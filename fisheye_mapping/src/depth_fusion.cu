// #pragma once
#include <vector>
#include <vector_types.h>
#include <cuda_toolkit/helper_math.h>
#include <rmd/device_image.cuh>
#include <rmd/texture_memory.cuh>
#include <rmd/match_parameter.cuh>
#include <rmd/camera_model/pinhole_camera.cuh>
#include <rmd/camera_model/fisheye_param.cuh>
// #include <rmd/feature_pixel.cuh>
#include <rmd/device_linear.cuh>
// #include "fisheye_project.cu"
#include <ctime>


//this file is uesd to fuse sequential depth map such that it is more stable

namespace rmd
{

//function declear here!
void depth_fuse_initialize(MatchParameter &match_parameter);
void depth_fuse(MatchParameter &match_parameter);

//new approach
//used for multibase line
__global__ void fuse_initialize_kernel(MatchParameter *match_parameter_devptr);
__global__ void fuse_transform(
  MatchParameter *match_parameter_devptr,
  const SE3<float> last_to_cur,
  DeviceImage<int> *transform_table_devptr);
__global__ void hole_filling(DeviceImage<int> *transform_table_devptr);
__global__ void fuse_currentmap(
  MatchParameter *match_parameter_devptr,
  DeviceImage<int> *transform_table_devptr);

__device__ __forceinline__ float normpdf(const float &x, const float &mu, const float &sigma_sq)
{
  return (expf(-(x-mu)*(x-mu) / (2.0f*sigma_sq))) * rsqrtf(2.0f*M_PI*sigma_sq);
}
// __device__ __forceinline__ bool is_goodpoint(const float4 &point_info)
// {
//   return (point_info.x /(point_info.x + point_info.y) > 0.6 && sqrtf(point_info.w)/point_info.z < 0.2);
// }
__device__ __forceinline__ bool is_goodpoint(const float4 &point_info)
{
  return (point_info.x /(point_info.x + point_info.y) > 0.6);
}
__device__ __forceinline__ bool is_badpoint(const float4 &point_info)
{
  return point_info.x /(point_info.x + point_info.y) < 0.45;
}

//function define here
void depth_fuse_initialize(MatchParameter &match_parameter)
{
  const int depth_width = match_parameter.predepth_fuse_hostptr->width;
  const int depth_height = match_parameter.predepth_fuse_hostptr->height;

  std::clock_t start = std::clock();
  dim3 fuse_block;
  dim3 fuse_grid;
  fuse_block.x = 32;
  fuse_block.y = 32;
  fuse_grid.x = (depth_width + fuse_block.x - 1) / fuse_block.x;
  fuse_grid.y = (depth_height + fuse_block.y - 1) / fuse_block.y;

  fuse_initialize_kernel<<<fuse_grid, fuse_block>>>(match_parameter.dev_ptr);
  cudaDeviceSynchronize();
  printf("  initialize the depth fuse cost %f ms\n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);
}

void depth_fuse(MatchParameter &match_parameter)
{
  const int depth_width = match_parameter.predepth_fuse_hostptr->width;
  const int depth_height = match_parameter.predepth_fuse_hostptr->height;

  dim3 fuse_block;
  dim3 fuse_grid;
  fuse_block.x = 32;
  fuse_block.y = 32;
  fuse_grid.x = (depth_width + fuse_block.x - 1) / fuse_block.x;
  fuse_grid.y = (depth_height + fuse_block.y - 1) / fuse_block.y;

  //first project the previouse depthfuse into current frame
  std::clock_t start = std::clock();
  DeviceImage<int> propogate_table(720, 720);
  propogate_table.zero();
  const SE3<float> last_to_cur
    = match_parameter.current_worldpose * (match_parameter.previous_worldpose).inv();
  fuse_transform<<<fuse_grid, fuse_block>>>(match_parameter.dev_ptr, last_to_cur, propogate_table.dev_ptr);
  cudaDeviceSynchronize();
  printf("  fuse transform cost %f ms\n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);
  
  start = std::clock();
  hole_filling<<<fuse_grid, fuse_block>>>(propogate_table.dev_ptr);
  cudaDeviceSynchronize();
  printf("  hole filling cost %f ms\n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);

  start = std::clock();
  fuse_currentmap<<<fuse_grid, fuse_block>>>(match_parameter.dev_ptr, propogate_table.dev_ptr);
  cudaDeviceSynchronize();
  printf("  fuse depth cost %f ms\n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);

  *match_parameter.predepth_fuse_hostptr = *match_parameter.curdepth_fuse_hostptr;
  *match_parameter.prefeature_age_hostptr = *match_parameter.curfeature_age_hostptr;
  match_parameter.curdepth_fuse_hostptr->zero();
}

__global__ void fuse_initialize_kernel(MatchParameter *match_parameter_devptr)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  DeviceImage<float4> *predepth_fuse_devptr = match_parameter_devptr->predepth_fuse_devptr;

  const int width = predepth_fuse_devptr->width;
  const int height = predepth_fuse_devptr->height;
  if(x >= width || y >= height)
    return;

  const float2 point = make_float2(x - width/2, y - height/2);
  const float r_2 = point.x * point.x + point.y * point.y;

  //129600 = 360*360;
  if(r_2 > 129600)
    return;

  //initial as a: 10, b:10, average depth 1.0m, sigma^2 = (50/6)^(50/6)
  predepth_fuse_devptr->atXY(x,y) = make_float4(10.0f, 10.0f, 1.0f, 70.0f);
}

__global__ void fuse_transform(
  MatchParameter *match_parameter_devptr,
  const SE3<float> last_to_cur,
  DeviceImage<int> *transform_table_devptr)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int index = x + y * 720;

  DeviceImage<float4> *predepth_fuse_devptr = match_parameter_devptr->predepth_fuse_devptr;

  const int width = predepth_fuse_devptr->width;
  const int height = predepth_fuse_devptr->height;

  if(x >= width || y >= height)
    return;

  const float2 point = make_float2(x - width/2, y - height/2);
  const float r = length(point);

  if(r > 360)
    return;

  const float phi_rad = atan2f(point.y, point.x);
  const float theta_rad = r * 0.0043633; // 0.0043633 = 1 / 4 / 180 * pi
  const float3 dir = make_float3(sinf(theta_rad) * cosf(phi_rad), sinf(theta_rad) * sinf(phi_rad), cosf(theta_rad));

  float4 pixel_info = predepth_fuse_devptr->atXY(x,y);
  float3 projected = last_to_cur * (dir * pixel_info.z);
  float new_depth = length(projected);
  float theta = acosf(projected.z / new_depth);
  float phi = atan2f(projected.y, projected.x);

  pixel_info.z = new_depth;
  pixel_info.w += 0.01;

  float r_ = theta * 229.183;
  const int projecte_x = r_ * cosf(phi) + width  / 2;
  const int projecte_y = r_ * sinf(phi) + height / 2;

  //projected out of the image
  if(r_ > 360 || projecte_x >= width || projecte_x < 0 || projecte_y >= height || projecte_y < 0)
    return;

  int *check_ptr = &(transform_table_devptr->atXY(projecte_x, projecte_y));
  int expect_i = 0;
  int actual_i;
  bool finish_job = false;
  while(!finish_job)
  {
    actual_i = atomicCAS(check_ptr, expect_i, index);
    if(actual_i != expect_i)
    {
      int now_x = actual_i % 720;
      int now_y = actual_i / 720;
      float now_d = (predepth_fuse_devptr->atXY(now_x, now_y)).z;
      if(now_d < new_depth)
        finish_job = true;
    }
    else
    {
      finish_job = true;
    }
    expect_i = actual_i;
  }

  predepth_fuse_devptr->atXY(x,y) = pixel_info;
}

__global__ void hole_filling(DeviceImage<int> *transform_table_devptr)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int width = 720;
  const int height = 720;
  const float2 point = make_float2(x - width/2, y - height/2);
  const float r_2 = point.x * point.x + point.y * point.y;

  if(r_2 > 129600 || x >= width || y >= height)
    return;

  const int transform_i = transform_table_devptr->atXY(x,y);

  if(transform_i == 0)
    return;

  for(int i = -1; i <= 1; i++)
  {
    for(int j = -1; j <= 1; j++)
    {
      int *neighbor = &(transform_table_devptr->atXY(x + j, y + i));
      atomicCAS(neighbor, 0, transform_i);
    }
  }
}

__global__ void fuse_currentmap(
  MatchParameter *match_parameter_devptr,
  DeviceImage<int> *transform_table_devptr)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;

  DeviceImage<float4> *predepth_fuse_devptr = match_parameter_devptr->predepth_fuse_devptr;
  DeviceImage<float4> *curdepth_fuse_devptr = match_parameter_devptr->curdepth_fuse_devptr;

  const int width = 720;
  const int height = 720;
  const float2 point = make_float2(x - width/2, y - height/2);
  const float r_2 = point.x * point.x + point.y * point.y;

  if(r_2 > 129600 || x >= width || y >= height)
    return;

  const int transform_i = transform_table_devptr->atXY(x,y);
  const int transform_x = transform_i % 720;
  const int transform_y = transform_i / 720;

  float depth_estimate = match_parameter_devptr->depth_undistorted_devptr->atXY(x,y);
  float uncertianity = depth_estimate * depth_estimate * 0.01;
  if(depth_estimate <= 0.0f)
    return;

  float4 pixel_info;
  bool track_lost = false;

  if(transform_i != 0)
    pixel_info = predepth_fuse_devptr->atXY(transform_x, transform_y);
  else
  {
    track_lost = true;
    pixel_info = make_float4(10.0f, 10.0f, 1.0f, 70.0f);
  }

  if( fabs(depth_estimate - pixel_info.z) > 1.5f || is_badpoint(pixel_info))
  {
    track_lost = true;
    pixel_info = make_float4(10.0f, 10.0f, depth_estimate, uncertianity);
  }

  //orieigin info`
  float a = pixel_info.x;
  float b = pixel_info.y;
  float miu = pixel_info.z;
  float sigma_sq = pixel_info.w;

  float new_sq = uncertianity * sigma_sq / (uncertianity + sigma_sq);
  float new_miu = (depth_estimate * sigma_sq + miu * uncertianity) / (uncertianity + sigma_sq);
  float c1 = (a / (a+b)) * normpdf(depth_estimate, miu, uncertianity + sigma_sq);
  float c2 = (b / (a+b)) * 1 / 50.0f;

  const float norm_const = c1 + c2;
  c1 = c1 / norm_const;
  c2 = c2 / norm_const;
  const float f = c1 * ((a + 1.0f) / (a + b + 1.0f)) + c2 *(a / (a + b + 1.0f));
  const float e = c1 * (( (a + 1.0f)*(a + 2.0f)) / ((a + b + 1.0f) * (a + b + 2.0f))) +
                  c2 *(a*(a + 1.0f) / ((a + b + 1.0f) * (a + b + 2.0f)));

  const float mu_prime = c1 * new_miu + c2 * miu;
  const float sigma_prime = c1 * (new_sq + new_miu * new_miu) + c2 * (sigma_sq + miu * miu) - mu_prime * mu_prime;
  const float a_prime = ( e - f ) / ( f - e/f );
  const float b_prime = a_prime * ( 1.0f - f ) / f;
  const float4 updated = make_float4(a_prime, b_prime, mu_prime, sigma_prime);

  __syncthreads();
  if(is_goodpoint(pixel_info))
    match_parameter_devptr->depth_undistorted_devptr->atXY(x,y) = mu_prime;
  else
    match_parameter_devptr->depth_undistorted_devptr->atXY(x,y) = -1.0f;
  curdepth_fuse_devptr->atXY(x,y) = updated;
  if(track_lost)
    match_parameter_devptr->curfeature_age_devptr->atXY(x/4,y/4) = 9;
}

}//namespace