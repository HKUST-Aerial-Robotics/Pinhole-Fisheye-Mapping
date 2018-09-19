// #pragma once
#include <vector>
#include <vector_types.h>
#include <cuda_toolkit/helper_math.h>
#include <rmd/device_image.cuh>
#include <rmd/texture_memory.cuh>
#include <rmd/match_parameter.cuh>
#include <rmd/camera_model/pinhole_camera.cuh>
#include <rmd/camera_model/fisheye_param.cuh>
// #include <rmd/grid_sample.cuh>
// #include <rmd/pixel_disparity.cuh>
// #include <rmd/feature_pixel.cuh>
#include <rmd/pixel_cost.cuh>
#include <rmd/device_linear.cuh>
// #include "tv_refine.cu"
// #include "curvature_refine.cu"
#include "hbp.cu"
#include "fisheye_project.cu"
// #include "bounded_tv.cu"
#include <ctime>


//this file is uesd to track support points using lbp

namespace rmd
{

//function declear here!
void match_features (MatchParameter &match_parameter, bool add_keyframe);

//new approach
//used for multibase line
__global__ void feature_age_project(  MatchParameter *match_parameter_devptr,
                                      SE3<float> last_to_current,
                                      bool add_keyframe);

//used for tracked feature
__global__ void prior_to_cost(  DeviceImage<PIXEL_COST> *cost_map_devptr,
                                DeviceLinear<float4> *features_devptr,
                                MatchParameter *match_parameter_devptr);

//used for feature aggregate
__global__ void feature_cost_aggregate( MatchParameter *match_parameter_devptr,
                                        DeviceImage<PIXEL_COST> *feature_cost_devptr);
__global__ void feature_depth_initial(  DeviceImage<PIXEL_COST> *feature_cost_devptr,
                                        DeviceImage<int> *depth_devptr,
                                        DeviceImage<int2> *depth_range_devptr,
                                        DeviceImage<float> *depth_uncertianity_devptr);

__global__ void depth_interpolate(  MatchParameter *match_parameter_devptr,
                                    DeviceImage<float> *featuredepth_devptr);
void generate_cloud_with_intensity( const SE3<float> &cur_to_world,
                                    const DeviceImage<float> &undistorted_ref,
                                    const DeviceImage<float> &undistorted_depth,
                                    DeviceImage<float4> &pub_cloud);
__global__ void cloud_with_intensity_kernel(
          const DeviceImage<float> *undistorted_depth_devptr,
          const DeviceImage<float> *undistorted_ref_devptr,
          DeviceImage<float4> *cloud_devptr,
          const SE3<float> cur_to_world);

__device__ __forceinline__ bool in_range(
                                  const int2 &check_point,
                                  const int &x_min,
                                  const int &y_min,
                                  const int &x_max,
                                  const int &y_max)
{
  return (check_point.x >= x_min && check_point.x < x_max && check_point.y >= y_min && check_point.y < y_max);
}

//debug
__global__ void draw_keyframe(  DeviceImage<float> *depth_ptr, 
                                DeviceImage<int> *dep_devptr,
                                int step)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int width = dep_devptr->width;
  const int height = dep_devptr->height;
  if(x >= width || y >= height)
    return;

  float value;

  if(is_vaild(step*x, step*y))
    value = dep_devptr->atXY(x,y);
  else
    value = 0;

  for(int i = 0; i < step; i++)
  {
    for(int j = 0; j < step; j++)
    {
      if(x * step + i < 752 && y * step + j< 480)
        depth_ptr->atXY(x * step + i, y * step + j) = value;
    }
  }
}

__global__ void change_debug(DeviceImage<float> *feature_devptr)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int width = feature_devptr->width;
  const int height = feature_devptr->height;

  if(x >= width || y >= height)
    return;

  feature_devptr->atXY(x,y) = (float)x + (float)y;
}

__global__ void draw_keyframe(  DeviceImage<float> *depth_ptr, 
                                DeviceImage<float> *other_devptr,
                                int step)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int width = other_devptr->width;
  const int height = other_devptr->height;
  if(x >= width || y >= height)
    return;
  else
  {
    for(int i = 0; i < step; i++)
    {
      for(int j = 0; j < step; j++)
      {
        if(x * step + i < 752 && y * step + j < 480)
          depth_ptr->atXY(x * step + i, y * step + j) = 1.0f / (other_devptr->atXY(x,y) * 0.0314 + 0.02);
          // depth_ptr->atXY(x * step + i, y * step + j) = other_devptr->atXY(x,y);
      }
    }
  }
}
__global__ void draw_image( DeviceImage<float> *depth_ptr, 
                            DeviceImage<float> *other_devptr,
                            int step)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int width = other_devptr->width;
  const int height = other_devptr->height;
  if(x >= width || y >= height)
    return;
  else
  {
    if(x  < 752 && y < 480)
      depth_ptr->atXY(x, y) = other_devptr->atXY(x,y);
  }
}
// __global__ void sample_grid_check( DeviceImage<GRID_SAMPLE> *sample_devptr);

//function define here
void match_features (MatchParameter &match_parameter, bool add_keyframe)
{
  const int image_width = match_parameter.image_width;
  const int image_height = match_parameter.image_height;
  const int keypoint_step = match_parameter.keypoint_step;
  const int keypoints_width = ceil((float)image_width / (float)keypoint_step);
  const int keypoints_height = ceil((float)image_height / (float)keypoint_step);

  printf("  select new features!\n");
  dim3 features_block;
  dim3 features_grid;
  features_block.x = 32;
  features_block.y = 32;
  features_grid.x = (keypoints_width + features_block.x - 1) / features_block.x;
  features_grid.y = (keypoints_height + features_block.y - 1) / features_block.y;

  DeviceImage<PIXEL_COST> feature_cost(keypoints_width, keypoints_height);
  DeviceImage<float> feature_depth(keypoints_width, keypoints_height);

  std::clock_t start = std::clock();
  const SE3<float> last_to_cur
      = match_parameter.current_worldpose * (match_parameter.previous_worldpose).inv();
  match_parameter.curfeature_age_hostptr->zero();
  feature_age_project<<<features_grid, features_block>>>(
    match_parameter.dev_ptr,
    last_to_cur,
    add_keyframe);
  printf("  feature age project cost %f ms\n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);
  // DeviceImage<int2> feature_depth_range(keypoints_width, keypoints_height);
  // DeviceImage<float> feature_certainity(keypoints_width, keypoints_height);

  start = std::clock();
  feature_cost.zero();
  dim3 cost_block;
  dim3 cost_grid;
  cost_block.x = DEPTH_NUM;
  cost_block.y = 10;
  cost_grid.x = keypoints_width;
  cost_grid.y = keypoints_height;
  feature_cost_aggregate<<<cost_grid, cost_block>>>
              ( match_parameter.dev_ptr,
                feature_cost.dev_ptr);
  cudaDeviceSynchronize();
  printf("  feature cost aggregaton cost %f ms\n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);
  
  start = std::clock();
  hbp(feature_cost, feature_depth);
  printf("  hbp cost %f ms\n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);

  start = std::clock();
  dim3 image_block;
  dim3 image_grid;
  image_block.x = 32;
  image_block.y = 32;
  image_grid.x = (image_width + image_block.x - 1) / image_block.x;
  image_grid.y = (image_height + image_block.y - 1) / image_block.y;
  depth_interpolate<<<image_grid, image_block>>>
      ( match_parameter.dev_ptr,
        feature_depth.dev_ptr);
  cudaDeviceSynchronize();
  printf("  depth interpolate cost %f ms\n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);
}


__global__ void feature_age_project(  MatchParameter *match_parameter_devptr,
                                      SE3<float> last_to_current,
                                      bool add_keyframe)
{
  const int feature_x = threadIdx.x + blockIdx.x * blockDim.x;
  const int feature_y = threadIdx.y + blockIdx.y * blockDim.y;
  const int width = match_parameter_devptr->image_width;
  const int height = match_parameter_devptr->image_height;
  const int keypoint_step = match_parameter_devptr->keypoint_step;
  DeviceImage<int> *predict_age_devptr = match_parameter_devptr->curfeature_age_devptr;

  const int x = feature_x * keypoint_step;
  const int y = feature_y * keypoint_step;

  if(x >= width || y >= height)
    return;

  const int last_age = match_parameter_devptr->prefeature_age_devptr->atXY(feature_x, feature_y);
  const FisheyeParam fisheye_param = match_parameter_devptr->fisheye_para;
  const DeviceImage<float> *last_depth_devptr = match_parameter_devptr->previous_depth_map_devptr;
  const float depth = last_depth_devptr->atXY(feature_x, feature_y);

  if(depth < 0.0f)
    return;
  
  float3 feature_dir = cam2world(make_float2(x, y), fisheye_param);
  float2 project_pixel = world2cam(last_to_current * (feature_dir * depth), fisheye_param);

  int2 project_int = make_int2( (int)(project_pixel.x), (int)(project_pixel.y));

  if (!is_vaild(project_int.x, project_int.y))
    return;

  int next_age = last_age;
  if(add_keyframe)
    next_age ++;
  predict_age_devptr->atXY((int)(project_pixel.x/keypoint_step), (int)(project_pixel.y/keypoint_step)) = next_age;
}

__global__ void image_depth_initial(  MatchParameter *match_parameter_devptr,
                                      DeviceImage<PIXEL_COST> *image_cost_devptr,
                                      DeviceImage<float> *depth_devptr,
                                      DeviceImage<float> *depth_cost_devptr)
{
  const int x = blockIdx.x;
  const int y = blockIdx.y;
  if( !is_vaild(x,y) )
  {
    depth_devptr->atXY(x,y) = -1.0f;
    return;
  }
  const int depth_id = threadIdx.x;
  const float start_invdepth = 1.0f / match_parameter_devptr->max_match_dist;
  const float inv_depth_step = match_parameter_devptr->inv_depth_step;

  __shared__ float cost[DEPTH_NUM];
  __shared__ float min_cost[DEPTH_NUM];
  __shared__ int min_index[DEPTH_NUM];

  cost[depth_id] = (image_cost_devptr->atXY(x,y)).get_cost(depth_id);
  cost[depth_id] = cost[depth_id] > 0.0f ? cost[depth_id] : 9999999.0f;
  min_cost[depth_id] = cost[depth_id];
  min_index[depth_id] = depth_id;

  cost[depth_id + DEPTH_NUM/2] = (image_cost_devptr->atXY(x,y)).get_cost(depth_id + DEPTH_NUM/2);
  cost[depth_id + DEPTH_NUM/2] = cost[depth_id + DEPTH_NUM/2] > 0.0f ? cost[depth_id + DEPTH_NUM/2] : 9999999.0f;
  min_cost[depth_id + DEPTH_NUM/2] = cost[depth_id + DEPTH_NUM/2];
  min_index[depth_id + DEPTH_NUM/2] = depth_id + DEPTH_NUM/2;
  __syncthreads();

  for(int i = DEPTH_NUM / 2; i > 0; i = i / 2)
  {
    if(depth_id < i && min_cost[depth_id + i] < min_cost[depth_id])
    {
      min_cost[depth_id] = min_cost[depth_id + i];
      min_index[depth_id] = min_index[depth_id + i];
    }
    __syncthreads();
  }

  if(depth_id == 0 && min_cost[0] < 9999998.0f)
  {
    if(min_index[0] > 0 && min_index[0] < DEPTH_NUM - 1)
    {
      float cost_pre = cost[min_index[0] - 1];
      float cost_post = cost[min_index[0] + 1];
      float a = cost_pre - 2.0f * min_cost[0] + cost_post;
      float b = - cost_pre + cost_post;
      float subpixel_idx = (float) min_index[0] - b / (2.0f * a);
      depth_devptr->atXY(x,y) = 1.0f / (start_invdepth + subpixel_idx * inv_depth_step);
    }
    else
    {
      depth_devptr->atXY(x,y) = 1.0f / (start_invdepth + min_index[0] * inv_depth_step);
    }
    depth_cost_devptr->atXY(x,y) = min_cost[0];
  }
}

__global__ void feature_depth_initial(  DeviceImage<PIXEL_COST> *feature_cost_devptr,
                                        DeviceImage<int> *depth_devptr,
                                        DeviceImage<int2> *depth_range_devptr,
                                        DeviceImage<float> *depth_certianity_devptr)
{
  const int depth_id = threadIdx.x;
  const int x = blockIdx.x;
  const int y = blockIdx.y;

  __shared__ float cost[DEPTH_NUM];
  __shared__ int min_index[DEPTH_NUM];
  __shared__ float min_cost[DEPTH_NUM];
  __shared__ int threshold_min[DEPTH_NUM];
  __shared__ int threshold_max[DEPTH_NUM];

  min_cost[depth_id] = cost[depth_id] = (feature_cost_devptr->atXY(x,y)).get_cost(depth_id);
  min_index[depth_id] = threshold_min[depth_id] = threshold_max[depth_id] = depth_id;
  min_cost[depth_id + DEPTH_NUM / 2] = cost[depth_id + DEPTH_NUM / 2] = (feature_cost_devptr->atXY(x,y)).get_cost(depth_id + DEPTH_NUM / 2);
  min_index[depth_id + DEPTH_NUM / 2] = threshold_min[depth_id + DEPTH_NUM / 2] = threshold_max[depth_id + DEPTH_NUM / 2] = depth_id  + DEPTH_NUM / 2;
  __syncthreads();

  for(int i = DEPTH_NUM / 2; i > 0; i /= 2)
  {
    if(depth_id < i && min_cost[depth_id + i] < min_cost[depth_id])
    {
      min_cost[depth_id] = min_cost[depth_id + i];
      min_index[depth_id] = min_index[depth_id + i];
    }
    __syncthreads();
  }

  float min_depth_cost = min_cost[0];
  float uncertainity_threshold = fmaxf(min_depth_cost * 1.1, 0.3f);

  if(cost[depth_id] > uncertainity_threshold && cost[depth_id] < 100000.0f)
  {
    threshold_min[depth_id] = DEPTH_NUM + 1;
    threshold_max[depth_id] = - 1;
  }
  if(cost[depth_id + DEPTH_NUM / 2] > uncertainity_threshold && cost[depth_id + DEPTH_NUM / 2] < 100000.0f)
  {
    threshold_min[depth_id + DEPTH_NUM / 2] = DEPTH_NUM + 1;
    threshold_max[depth_id + DEPTH_NUM / 2] = - 1;
  }
  __syncthreads();
  for(int i = DEPTH_NUM / 2; i > 0; i /= 2)
  {
    if(depth_id < i && threshold_min[depth_id + i] <= threshold_min[depth_id])
    {
      threshold_min[depth_id] = threshold_min[depth_id + i];
    }
    if(depth_id < i && threshold_max[depth_id + i] > threshold_max[depth_id])
    {
      threshold_max[depth_id] = threshold_max[depth_id + i];
    }
    __syncthreads();
  }

  if(depth_id == 0)
  {
    depth_devptr->atXY(x,y) = min_index[0];
    depth_range_devptr->atXY(x,y) =  make_int2(threshold_min[0], threshold_max[0]);
    float range = (threshold_max[0] - threshold_min[0]) / 64.0f;
    depth_certianity_devptr->atXY(x,y) = exp( - 2.0f * range * range);
    // depth_certianity_devptr->atXY(x,y) = 1.0f;
  }
}

__global__ void feature_cost_aggregate( MatchParameter *match_parameter_devptr,
                                        DeviceImage<PIXEL_COST> *feature_cost_devptr)
{
  const int depth_id = threadIdx.x;
  const int frame_id = threadIdx.y;
  const int image_width = match_parameter_devptr->image_width;
  const int image_height = match_parameter_devptr->image_height;
  const int keypoint_step = match_parameter_devptr->keypoint_step;
  const FisheyeParam fisheye_param = match_parameter_devptr->fisheye_para;
  const float start_invdepth = 1.0f / match_parameter_devptr->max_match_dist;
  const float inv_depth_step = match_parameter_devptr->inv_depth_step;
  const int feature_x = blockIdx.x * keypoint_step;
  const int feature_y = blockIdx.y * keypoint_step;

  if( ! is_vaild(feature_x, feature_y, fisheye_param) )
    return;

  DeviceImage<int> *feature_age_devptr = match_parameter_devptr->curfeature_age_devptr;
  float feature_age = feature_age_devptr->atXY(blockIdx.x, blockIdx.y);
  feature_age = feature_age > 29 ? 29 : feature_age;
  feature_age = feature_age < 10 ? 10 : feature_age;
  const int check_id = (int)(feature_age / (float)blockDim.y * (float)frame_id);
  // const int check_id = frame_id;

  const DeviceImage<float> *ref_dev = match_parameter_devptr->keyframe_devptr[check_id];
  const SE3<float> cur_to_ref = match_parameter_devptr->keyframe_worldpose[check_id] * (match_parameter_devptr->current_worldpose).inv();
  float3 feature_dir = cam2world(make_float2(feature_x, feature_y), fisheye_param);
  float featue_patch[3][3];
  for(int j = 0; j < 3; j++)
  {
    for(int i = 0; i < 3; i++)
    {
      featue_patch[i][j] = tex2D(curr_img_tex, feature_x + i - 1 + 0.5f, feature_y + j - 1 + 0.5f);
    }
  }

  __shared__ float cost_per_frame[DEPTH_NUM][10];
  __shared__ float get_value[DEPTH_NUM][10];

  float check_depth = 1.0f / (start_invdepth + depth_id * inv_depth_step);
  float2 check_pixel = world2cam(cur_to_ref * (feature_dir * check_depth), fisheye_param);
  int2 check_int = make_int2((int)check_pixel.x, (int)check_pixel.y);
  if( is_vaild(check_int, fisheye_param) )
  {
    float cost(0.0f);
    for(int shift_j = 0; shift_j < 3; shift_j ++)
    {
      for(int shift_i = 0; shift_i < 3; shift_i ++)
        cost += fabs( featue_patch[shift_i][shift_j] - ref_dev->atXY(check_int.x + shift_i - 1, check_int.y + shift_j - 1));
    }
    cost_per_frame[depth_id][frame_id] = cost;
    get_value[depth_id][frame_id] = 1.0f;
  }
  else
  {
    cost_per_frame[depth_id][frame_id] = 0.0f;
    get_value[depth_id][frame_id] = 0.0f;
  }

  __syncthreads();

  for(int r = 8; r > 0; r /= 2)
  {
    if(frame_id + r < blockDim.y && frame_id < r)
    {
      cost_per_frame[depth_id][frame_id] += cost_per_frame[depth_id][frame_id + r];
      get_value[depth_id][frame_id] += get_value[depth_id][frame_id + r];
    }
    __syncthreads();
  }

  if(frame_id == 0)
  {
    if(get_value[depth_id][0] != 0.0f)
    {
      (feature_cost_devptr->atXY(blockIdx.x, blockIdx.y)).set_cost(depth_id, cost_per_frame[depth_id][0] / get_value[depth_id][0]);
    }
    else
      (feature_cost_devptr->atXY(blockIdx.x, blockIdx.y)).set_cost(depth_id, 100000.0f);
  }
}

__global__ void depth_interpolate(  MatchParameter *match_parameter_devptr,
                                    DeviceImage<float> *featuredepth_devptr)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const FisheyeParam fisheye_param = match_parameter_devptr->fisheye_para;
  DeviceImage<float> *depth_output_devptr = match_parameter_devptr->depth_output_devptr;

  if( x >= 752 || y >= 480)
    return;

  if(! is_vaild(x, y, fisheye_param) )
  {
    depth_output_devptr->atXY(x, y) = -1.0f;
    return;
  }

  const int feature_width = featuredepth_devptr->width;
  const int feature_height = featuredepth_devptr->height;
  const float this_intensity = tex2D(curr_img_tex, x + 0.5f, y + 0.5f);
  const int keypoint_step = match_parameter_devptr->keypoint_step;
  const float start_invdepth = 1.0f / match_parameter_devptr->max_match_dist;
  const float inv_depth_step = match_parameter_devptr->inv_depth_step;

  float this_feature_weight;
  float this_feature_depth;
  float weight_sum = 0.0f;
  float sparital_factor = 1.0f;
  float intensity_factor = 0.5f;

  float interpolated_depth = 0.0f;

  for(int i = 0; i < 5; i++)
  {
    for(int j = 0; j < 5; j++)
    {
      int check_x = x / keypoint_step + i - 2;
      int check_y = y / keypoint_step + j - 2;

      if(check_x < 0 || check_x >= feature_width || check_y < 0 || check_y >= feature_height)
      {
        this_feature_weight = 0.0f;
        this_feature_depth = 0.0f;
      }
      else
      {
        this_feature_depth = featuredepth_devptr->atXY(check_x, check_y);
        float intensity_diff = this_intensity - tex2D(curr_img_tex, check_x * keypoint_step + 0.5f, check_y * keypoint_step + 0.5f);
        float2 sparital_diff = make_float2(x - check_x * keypoint_step, y - check_y * keypoint_step);
        float sparital_weight = sparital_diff.x * sparital_diff.x + sparital_diff.y * sparital_diff.y;
        float intensity_weight = intensity_diff * intensity_diff;
        this_feature_weight = expf( - sparital_weight / sparital_factor - intensity_weight / intensity_factor);
      }

      interpolated_depth += this_feature_weight * this_feature_depth;
      weight_sum += this_feature_weight;
    }
  }

  //normalize the weight
  if(weight_sum > 0.0f)
  {
    interpolated_depth = interpolated_depth / weight_sum;
    depth_output_devptr->atXY(x, y) = 1.0f / (interpolated_depth * inv_depth_step + start_invdepth);
  }
  else
  {
    depth_output_devptr->atXY(x, y) = - 1.0f;
  }
}

void generate_cloud_with_intensity( const SE3<float> &cur_to_world,
                                    const DeviceImage<float> &undistorted_ref,
                                    const DeviceImage<float> &undistorted_depth,
                                    DeviceImage<float4> &pub_cloud)
{
  const int image_width = undistorted_ref.width;
  const int image_height = undistorted_ref.height;
  dim3 image_block;
  dim3 image_grid;
  image_block.x = 32;
  image_block.y = 32;
  image_grid.x = (image_width + image_block.x - 1) / image_block.x;
  image_grid.y = (image_height + image_block.y - 1) / image_block.y;

  cloud_with_intensity_kernel<<<image_grid, image_block>>>(
    undistorted_depth.dev_ptr,
    undistorted_ref.dev_ptr,
    pub_cloud.dev_ptr,
    cur_to_world);
  cudaDeviceSynchronize();
}

__global__ void cloud_with_intensity_kernel(
          const DeviceImage<float> *undistorted_depth_devptr,
          const DeviceImage<float> *undistorted_ref_devptr,
          DeviceImage<float4> *cloud_devptr,
          const SE3<float> cur_to_world)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int width = undistorted_depth_devptr->width;
  const int height = undistorted_depth_devptr->height;

  if(x >= width || y >= height)
    return;

  float depth = undistorted_depth_devptr->atXY(x,y);

  if(depth <= 0.0f)
  {
    cloud_devptr->atXY(x,y) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    return;
  }

  float2 point = make_float2(x - width/2, y - height/2);
  float theta = length(point) * 0.0043633; // 0.0043633 = 1 / 4 / 180 * pi
  float phi = atan2f(point.y, point.x);
  float3 dir = make_float3(sinf(theta) * cosf(phi), sinf(theta) * sinf(phi), cosf(theta));
  float3 cloud = cur_to_world * (dir * depth);
  float intensity = undistorted_ref_devptr->atXY(x,y);

  cloud_devptr->atXY(x, y) = make_float4(cloud.x, cloud.y, cloud.z, intensity);
}

__global__ void prior_to_cost(  DeviceImage<PIXEL_COST> *cost_map_devptr,
                                DeviceLinear<float4> *features_devptr,
                                MatchParameter *match_parameter_devptr)
{
  const int feature_index = threadIdx.x + blockDim.x * blockIdx.x;
  const int neighbor_index = threadIdx.y;
  const int feature_size = features_devptr->length;

  if(feature_index >= feature_size)
    return;

  const int cost_width = cost_map_devptr->width;
  const int cost_height = cost_map_devptr->height;

  float4 feature_info = features_devptr->at(feature_index);
  const int feature_x = feature_info.x / 4;
  const int feature_y = feature_info.y / 4;

  if(feature_x <= 0 || feature_x >= cost_width - 1 || feature_y <= 0 || feature_y >= cost_height - 1)
    return;

  __shared__ float sum_weight;
  sum_weight = 0.0f;
  __syncthreads();

  const int this_x = feature_x + neighbor_index / 2;
  const int this_y = feature_y + neighbor_index % 2;
  const float feature_intensity = tex2D(curr_img_tex, feature_info.x + 0.5f, feature_info.y + 0.5f);
  const float this_intensity = tex2D(curr_img_tex, this_x * 4 + 0.5f, this_y * 4 + 0.5f);

  float2 sparital_diff = make_float2(this_x * 4, this_y * 4) - make_float2(feature_info.x, feature_info.y);
  float sparital_weight = sparital_diff.x * sparital_diff.x + sparital_diff.y * sparital_diff.y;
  float intensity_diff = (this_intensity - feature_intensity);
  float intensity_weight = intensity_diff * intensity_diff;
  float sparital_factor = 1.0f;
  float intensity_factor = 0.1f;
  float weigth = expf( - sparital_weight / sparital_factor - intensity_weight / intensity_factor);
  atomicAdd(&sum_weight, weigth);
  __syncthreads();

  weigth = weigth / sum_weight;
  float inv_depth = feature_info.z;
  float inv_max_error = fmaxf(2 * feature_info.w, 2);
  float sigma_sq = feature_info.w * feature_info.w;
  float prior_factor = 0.2f;
  const float start_invdepth = 1.0f / match_parameter_devptr->max_match_dist;
  const float inv_depth_step = match_parameter_devptr->inv_depth_step;
  for(int i = 0; i < DEPTH_NUM; i++)
  {
    float inv_diff = fabs(start_invdepth + i * inv_depth_step - inv_depth);
    inv_diff = inv_depth > inv_max_error ? inv_max_error : inv_depth;
    float prior_cost = weigth * inv_diff * inv_diff / sigma_sq / prior_factor;
    (cost_map_devptr->atXY(this_x, this_y)).add_cost(i, prior_cost);
  }
  // printf(" the error inv is %f  cout for %f desparity.\n", feature_info.w, feature_info.w / inv_depth_step);
}

}//namespace