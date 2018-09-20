// #pragma once
#include <vector>
#include <vector_types.h>
#include <cuda_toolkit/helper_math.h>
#include <rmd/device_image.cuh>
#include <rmd/texture_memory.cuh>
#include <rmd/match_parameter.cuh>
#include <rmd/camera_model/pinhole_camera.cuh>
#include <rmd/pixel_cost.cuh>
#include <rmd/device_linear.cuh>
#include "hbp.cu"
// #include "bounded_tv.cu"
#include <ctime>


//this file is uesd to track support points using TV

namespace rmd
{

//function declear here!
void match_features ( MatchParameter &match_parameter, bool add_keyframe);

//new approach
//used for multibase line
__global__ void feature_age_project(  MatchParameter *match_parameter_devptr,
                                      DeviceImage<int> *predict_age_devptr,
                                      SE3<float> last_to_current,
                                      bool add_keyframe);

//used for feature aggregate
__global__ void feature_cost_aggregate( MatchParameter *match_parameter_devptr,
                                        DeviceImage<int> *feature_age_devptr,
                                        DeviceImage<PIXEL_COST> *feature_cost_devptr);
__global__ void feature_depth_initial(  DeviceImage<PIXEL_COST> *feature_cost_devptr,
                                        DeviceImage<int> *depth_devptr,
                                        DeviceImage<int2> *depth_range_devptr,
                                        DeviceImage<float> *depth_uncertianity_devptr);

// __global__ void initial_similarity( MatchParameter *match_parameter_devptr,
//                                     DeviceImage<float4> *similarity_devptr);

//depth upsampling
// __global__ void  initial_sample_grid( DeviceImage<float> *depth_devptr,
//                                       DeviceImage<GRID_SAMPLE> *sample_devptr);

// __global__ void image_cost_aggregate( MatchParameter *match_parameter_devptr,
//                                       DeviceImage<PIXEL_COST> *image_cost_devptr,
//                                       DeviceImage<GRID_SAMPLE> *sample_grid_devptr,
//                                       DeviceImage<float> *featuredepth_devptr,
//                                       int grid_step);
// __global__ void image_depth_initial(  MatchParameter *match_parameter_devptr,
//                                       DeviceImage<PIXEL_COST> *image_cost_devptr,
//                                       DeviceImage<float> *depth_devptr,
//                                       DeviceImage<float> *depth_certian_devptr);

//depth interpolate
__global__ void depth_interpolate(  MatchParameter *match_parameter_devptr,
                                    DeviceImage<float> *featuredepth_devptr,
                                    DeviceImage<float> *depth_devptr);

//for depth consist filter
__global__ void LR_check( MatchParameter *match_parameter_devptr, 
                          DeviceImage<float> *depth_cost_devptr,
                          DeviceImage<int> *predice_age_devptr,
                          const SE3<float> cur_to_last, 
                          const SE3<float> cur_to_last2);
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
                                DeviceImage<int> *dep_devptr)
{
  int step = 4;
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int width = dep_devptr->width;
  const int height = dep_devptr->height;
  if(x >= width || y >= height)
    return;
  else
  {
    for(int i = 0; i < step; i++)
    {
      for(int j = 0; j < step; j++)
      {
        if(x * step + i < 752 && y * step + j< 480)
          depth_ptr->atXY(x * step + i, y * step + j) = dep_devptr->atXY(x,y);
      }
    }
  }

}
__global__ void draw_keyframe(  DeviceImage<float> *depth_ptr, 
                                DeviceImage<float> *other_devptr)
{
  int step = 4;
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
        if(x * step + i < 752 && y * step + j< 480)
          // depth_ptr->atXY(x * step + i, y * step + j) = other_devptr->atXY(x,y);
          depth_ptr->atXY(x * step + i, y * step + j) = 1.0f / (0.0314 * other_devptr->atXY(x,y) + 0.02);
      }
    }
  }
}
__global__ void draw_image(  DeviceImage<float> *depth_ptr, 
                            DeviceImage<float> *other_devptr)
{
  int step = 1;
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
  features_block.x = 16;
  features_block.y = 16;
  features_grid.x = (keypoints_width + features_block.x - 1) / features_block.x;
  features_grid.y = (keypoints_height + features_block.y - 1) / features_block.y;

  std::clock_t start = std::clock();
  DeviceImage<PIXEL_COST> feature_cost(keypoints_width, keypoints_height);
  DeviceImage<float> feature_depth(keypoints_width, keypoints_height);
  DeviceImage<int> predict_age(keypoints_width, keypoints_height);
  predict_age.zero();

  const SE3<float> last_to_cur
      = match_parameter.current_worldpose * (match_parameter.previous_worldpose).inv();
  feature_age_project<<<features_grid, features_block>>>(
    match_parameter.dev_ptr,
    predict_age.dev_ptr,
    last_to_cur,
    add_keyframe);
  printf("  feature age project cost %f ms\n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);

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
                predict_age.dev_ptr,
                feature_cost.dev_ptr);
  cudaDeviceSynchronize();
  printf("  feature cost aggregaton cost %f ms\n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);

  start = std::clock();
  hbp(feature_cost, feature_depth);
  printf("  hbp cost %f ms\n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);

  // // guided sample level
  // const int sample_size = 8; //8x8 keypoints form a sample grid
  // const int grid_x = (keypoints_width + sample_size - 1) / sample_size;
  // const int grid_y = (keypoints_height + sample_size - 1) / sample_size;
  // //create grid
  // DeviceImage<GRID_SAMPLE> grid_sample(grid_x, grid_y);
  // DeviceImage<PIXEL_COST> full_cost_map(image_width, image_height);
  // grid_sample.zero();
  // full_cost_map.zero();
  // dim3 sample_block;
  // dim3 sample_grid;
  // sample_block.x = sample_size;
  // sample_block.y = sample_size;
  // sample_grid.x = (keypoints_width + sample_block.x - 1) / sample_block.x;
  // sample_grid.y = (keypoints_height + sample_block.y - 1) / sample_block.y;
  // start = std::clock();
  // initial_sample_grid<<<sample_grid, sample_block>>>(
  //             feature_depth.dev_ptr,
  //             grid_sample.dev_ptr);
  // cudaDeviceSynchronize();

  // start = std::clock();
  // dim3 image_sample_block;
  // dim3 image_sample_grid;
  // image_sample_block.x = 8;
  // image_sample_block.y = 8;
  // image_sample_block.z = 5;
  // image_sample_grid.x = (image_width + image_sample_block.x - 1) / image_sample_block.x;
  // image_sample_grid.y = (image_height + image_sample_block.y - 1) / image_sample_block.y;
  // image_cost_aggregate<<<image_sample_grid, image_sample_block>>>
  //             ( match_parameter.dev_ptr,
  //               full_cost_map.dev_ptr,
  //               grid_sample.dev_ptr,
  //               feature_depth.dev_ptr,
  //               sample_size * 4);
  // cudaDeviceSynchronize();
  // printf("  full cost aggregation 2 cost %f ms\n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);

  // start = std::clock();
  // dim3 image_filter_block;
  // dim3 image_filter_grid;
  // image_filter_block.x = 32;
  // image_filter_grid.x = image_width;
  // image_filter_grid.y = image_height;
  // DeviceImage<float> depth_cost(image_width, image_height);
  // depth_cost.zero();
  // image_depth_initial<<<image_filter_grid, image_filter_block>>>
  //           ( match_parameter.dev_ptr,
  //             full_cost_map.dev_ptr,
  //             match_parameter.depth_map_devptr,
  //             depth_cost.dev_ptr);
  // cudaDeviceSynchronize();
  // printf("  full image depth cost %f ms\n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);

  start = std::clock();
  dim3 image_block;
  dim3 image_grid;
  image_block.x = 8;
  image_block.y = 8;
  image_grid.x = (image_width + image_block.x - 1) / image_block.x;
  image_grid.y = (image_height + image_block.y - 1) / image_block.y;
  depth_interpolate<<<image_grid, image_block>>>
      ( match_parameter.dev_ptr,
        feature_depth.dev_ptr,
        match_parameter.depth_output_devptr);
  cudaDeviceSynchronize();
  printf("  depth interpolate cost %f ms\n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);

  // if(match_parameter.frame_nums < 13)
  // {
  //   *match_parameter.previous2_depth_map_hostptr = *match_parameter.previous_depth_map_hostptr; //the memory in device copys
  //   *match_parameter.previous_depth_map_hostptr = *match_parameter.depth_map_hostptr; ////the memory in device copys
  //   match_parameter.previous2_worldpose = match_parameter.previous_worldpose;
  //   match_parameter.previous_worldpose = match_parameter.current_worldpose;
  //   return;
  // }

  // start = std::clock();
  // const SE3<float> cur_to_last
  //       = match_parameter.previous_worldpose * (match_parameter.current_worldpose).inv();
  // const SE3<float> cur_to_last2
  //       = match_parameter.previous2_worldpose * (match_parameter.current_worldpose).inv();

  // LR_check<<<image_grid, image_block>>>
  //         ( match_parameter.dev_ptr,
  //           depth_cost.dev_ptr,
  //           predict_age.dev_ptr,
  //           cur_to_last,
  //           cur_to_last2);
  // cudaDeviceSynchronize();
  // printf("  LR check cost %f ms\n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);

  // *match_parameter.previous2_depth_map_hostptr = *match_parameter.previous_depth_map_hostptr;
  // *match_parameter.previous_depth_map_hostptr = *match_parameter.depth_map_hostptr;
  // *match_parameter.prefeature_age_hostptr = predict_age;
  // match_parameter.previous2_worldpose = match_parameter.previous_worldpose;
  // match_parameter.previous_worldpose = match_parameter.current_worldpose;
  // draw_keyframe<<<features_grid, features_block>>>(match_parameter.depth_output_devptr, predict_age.dev_ptr);
}

__global__ void feature_age_project(  MatchParameter *match_parameter_devptr,
                                      DeviceImage<int> *predict_age_devptr,
                                      SE3<float> last_to_current,
                                      bool add_keyframe)
{
  const int feature_x = threadIdx.x + blockIdx.x * blockDim.x;
  const int feature_y = threadIdx.y + blockIdx.y * blockDim.y;
  const int width = match_parameter_devptr->image_width;
  const int height = match_parameter_devptr->image_height;
  const int keypoint_step = match_parameter_devptr->keypoint_step;

  const int x = feature_x * keypoint_step;
  const int y = feature_y * keypoint_step;

  if(x >= width || y >= height)
    return;

  const int last_age = match_parameter_devptr->prefeature_age_devptr->atXY(feature_x, feature_y);
  const PinholeCamera *camera_devptr = match_parameter_devptr->camera_devptr;
  const DeviceImage<float> *last_depth_devptr = match_parameter_devptr->previous_depth_map_devptr;
  const float depth = last_depth_devptr->atXY(feature_x, feature_y);
  float3 feature_dir = normalize(camera_devptr->cam2world(make_float2(x, y)));
  float2 project_pixel = camera_devptr->world2cam(last_to_current * (feature_dir * depth));

  int2 project_int = make_int2( (int)(project_pixel.x), (int)(project_pixel.y));

  if( project_int.x < 0 || project_int.x >= width || project_int.y < 0 || project_int.y >= height)
    return;

  int next_age = add_keyframe ? (last_age + 1) : last_age;
  predict_age_devptr->atXY((int)(project_pixel.x/4), (int)(project_pixel.y/4)) = next_age;
}

// __global__ void initial_sample_grid( DeviceImage<float> *depth_devptr,
//                                      DeviceImage<GRID_SAMPLE> *sample_devptr)
// {
//   const int feature_x = threadIdx.x + blockIdx.x * blockDim.x;
//   const int feature_y = threadIdx.y + blockIdx.y * blockDim.y;
//   const int width = depth_devptr->width;
//   const int height = depth_devptr->height;
//   const int local_id = threadIdx.y * blockDim.x + threadIdx.x;

//   const int grid_width = sample_devptr->width;
//   const int grid_height = sample_devptr->height;

//   if(feature_x >= width || feature_y >= height)
//     return;

//   __shared__ bool need_sample[DEPTH_NUM];
//   if(local_id < DEPTH_NUM)
//     need_sample[local_id] = false;
//   __syncthreads();

//   float this_depth = depth_devptr->atXY(feature_x, feature_y);
//   need_sample[(int)this_depth] = true;
//   if((int)this_depth + 1 < DEPTH_NUM - 1)
//     need_sample[(int)this_depth + 1] = true;
//   __syncthreads();

//   if(local_id < DEPTH_NUM && need_sample[local_id])
//   {
//     for(int i = -1; i <= 1; i++)
//     {
//       for(int j = -1; j <= 1; j++)
//       {
//         int neighbor_x = blockIdx.x + i;
//         int neighbor_y = blockIdx.y + j;
//         if(in_range(make_int2(neighbor_x, neighbor_y), 0, 0, grid_width, grid_height))
//           (sample_devptr->atXY(neighbor_x, neighbor_y)).need_sample[local_id] = true;
//       }
//     }
//   }
// }

// __global__ void sample_grid_check( DeviceImage<GRID_SAMPLE> *sample_devptr)
// {
//   const int x = threadIdx.x + blockIdx.x * blockDim.x;
//   const int y = threadIdx.y + blockIdx.y * blockDim.y;
//   const int width = sample_devptr->width;
//   const int height = sample_devptr->height;
//   if(x >= width || y >= height)
//     return;
//   int max_index = -1;
//   int min_index = DEPTH_NUM;

//   for(int i = 0 ; i < 64; i++)
//   {
//     if( (sample_devptr->atXY(x,y)).need_sample[i] )
//     {
//       if(i < min_index)
//         min_index = i;
//       if(i > max_index)
//         max_index = i;
//     }
//   }
//   printf("grid (%d, %d) sample (%d -> %d)\n", x, y, min_index, max_index);
// }

// __global__ void image_depth_initial(  MatchParameter *match_parameter_devptr,
//                                       DeviceImage<PIXEL_COST> *image_cost_devptr,
//                                       DeviceImage<float> *depth_devptr,
//                                       DeviceImage<float> *depth_cost_devptr)
// {
//   const int x = blockIdx.x;
//   const int y = blockIdx.y;
//   const int depth_id = threadIdx.x;
//   const float start_invdepth = 1.0f / match_parameter_devptr->max_match_dist;
//   const float inv_depth_step = match_parameter_devptr->inv_depth_step;

//   __shared__ float cost[DEPTH_NUM];
//   __shared__ float min_cost[DEPTH_NUM];
//   __shared__ int min_index[DEPTH_NUM];

//   cost[depth_id] = (image_cost_devptr->atXY(x,y)).get_cost(depth_id);
//   cost[depth_id] = cost[depth_id] > 0.0f ? cost[depth_id] : 9999999.0f;
//   min_cost[depth_id] = cost[depth_id];
//   min_index[depth_id] = depth_id;

//   cost[depth_id + DEPTH_NUM/2] = (image_cost_devptr->atXY(x,y)).get_cost(depth_id + DEPTH_NUM/2);
//   cost[depth_id + DEPTH_NUM/2] = cost[depth_id + DEPTH_NUM/2] > 0.0f ? cost[depth_id + DEPTH_NUM/2] : 9999999.0f;
//   min_cost[depth_id + DEPTH_NUM/2] = cost[depth_id + DEPTH_NUM/2];
//   min_index[depth_id + DEPTH_NUM/2] = depth_id + DEPTH_NUM/2;
//   __syncthreads();

//   for(int i = DEPTH_NUM / 2; i > 0; i = i / 2)
//   {
//     if(depth_id < i && min_cost[depth_id + i] < min_cost[depth_id])
//     {
//       min_cost[depth_id] = min_cost[depth_id + i];
//       min_index[depth_id] = min_index[depth_id + i];
//     }
//     __syncthreads();
//   }

//   if(depth_id == 0 && min_cost[0] < 9999998.0f)
//   {
//     if(min_index[0] > 0 && min_index[0] < DEPTH_NUM - 1)
//     {
//       float cost_pre = cost[min_index[0] - 1];
//       float cost_post = cost[min_index[0] + 1];
//       float a = cost_pre - 2.0f * min_cost[0] + cost_post;
//       float b = - cost_pre + cost_post;
//       float subpixel_idx = (float) min_index[0] - b / (2.0f * a);
//       depth_devptr->atXY(x,y) = 1.0f / (start_invdepth + subpixel_idx * inv_depth_step);
//     }
//     else
//     {
//       depth_devptr->atXY(x,y) = 1.0f / (start_invdepth + min_index[0] * inv_depth_step);
//     }
//     depth_cost_devptr->atXY(x,y) = min_cost[0];
//   }
// }

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
    depth_certianity_devptr->atXY(x,y) = expf( - 2.0f * range * range);
    // depth_certianity_devptr->atXY(x,y) = 1.0f;
  }
}

__global__ void depth_interpolate(  MatchParameter *match_parameter_devptr,
                                    DeviceImage<float> *featuredepth_devptr,
                                    DeviceImage<float> *depth_devptr)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int image_width = depth_devptr->width;
  const int image_height = depth_devptr->height;

  if(x >= image_width || y >= image_height)
    return;

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

  for(int i = 0; i < 5; i ++)
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
  interpolated_depth = interpolated_depth / weight_sum;
  depth_devptr->atXY(x,y) = 1.0f / (interpolated_depth * inv_depth_step + start_invdepth);
}

// __global__ void image_cost_aggregate( MatchParameter *match_parameter_devptr,
//                                       DeviceImage<PIXEL_COST> *image_cost_devptr,
//                                       DeviceImage<GRID_SAMPLE> *sample_grid_devptr,
//                                       DeviceImage<float> *feature_depth,
//                                       int grid_step)
// {
//   const int frame_id = threadIdx.z;
//   const int frame_num = blockDim.z;
//   const int x = threadIdx.x + blockIdx.x * blockDim.x;
//   const int y = threadIdx.y + blockIdx.y * blockDim.y;
//   const int pixel_id = threadIdx.y * blockDim.x + threadIdx.x;
//   const int local_id = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
//   const int image_width = match_parameter_devptr->image_width;
//   const int image_height = match_parameter_devptr->image_height;

//   const PinholeCamera *camera_devptr = match_parameter_devptr->camera_devptr;
//   const float start_invdepth = 1.0f / match_parameter_devptr->max_match_dist;
//   const float inv_depth_step = match_parameter_devptr->inv_depth_step;

//   const int grid_x = x / grid_step;
//   const int grid_y = y / grid_step;
//   const int feature_x = x / 4;
//   const int feature_y = y / 4;
//   const int feature_width = feature_depth->width;
//   const int feature_height = feature_depth->height;


//   const DeviceImage<float> *ref_dev = match_parameter_devptr->keyframe_devptr[frame_id];
//   const SE3<float> cur_to_ref = match_parameter_devptr->keyframe_worldpose[frame_id] * (match_parameter_devptr->current_worldpose).inv();
//   float3 feature_dir = normalize(camera_devptr->cam2world(make_float2(x, y)));
//   float featue_patch[3][3];
//   for(int j = 0; j < 3; j++)
//   {
//     for(int i = 0; i < 3; i++)
//     {
//       featue_patch[i][j] = tex2D(curr_img_tex, x + i - 1 + 0.5f, y + j - 1 + 0.5f);
//     }
//   }

//   float neighbor_weight[25];
//   float neighbor_depth[25];
//   float norm_weight(0.0f);
//   int neighbor_num(0);

//   if(frame_id == 0)
//   {
//   for(int j = -2; j <= 2; j ++)
//   {
//     for(int i = -2; i <= 2; i++)
//     {
//       int check_x = feature_x + i;
//       int check_y = feature_y + j;
//       float sparital_factor = 1.0f;
//       float intensity_factor = 0.1f;
//       if(check_x >= 0 && check_x < feature_width && check_y >= 0 && check_y < feature_height)
//       {
//         float sparital_weight = (i * i + j * j);
//         float intensity_weight = featue_patch[1][1] - tex2D(curr_img_tex, check_x * 4 + 0.5f, check_y * 4 + 0.5f);
//         float depth = feature_depth->atXY(check_x, check_y);
//         intensity_weight = intensity_weight * intensity_weight;
//         neighbor_depth[neighbor_num] = depth;
//         neighbor_weight[neighbor_num] = expf( - sparital_weight / sparital_factor - intensity_weight / intensity_factor);
//         norm_weight += neighbor_weight[neighbor_num];
//         neighbor_num ++;
//       }
//     }
//   }
//   }

//   __shared__ float cost_per_frame[64][10];
//   __shared__ float get_value[64][10];
//   __shared__ bool need_sample[64];
//   __shared__ int need_sample_max[64];
//   __shared__ int need_sample_min[64];

//   if(local_id < 64)
//   {
//     need_sample[local_id] = (sample_grid_devptr->atXY(grid_x, grid_y)).need_sample[local_id];
//     if(need_sample[local_id])
//     {
//       need_sample_max[local_id] = local_id;
//       need_sample_min[local_id] = local_id;
//     }
//     else
//     {
//       need_sample_max[local_id] = -1;
//       need_sample_min[local_id] = DEPTH_NUM + 1;
//     }
//   }

//   __syncthreads();
//   for(int i = 64 / 2; i > 0; i = i / 2)
//   {
//     if(local_id < i && (need_sample_max[local_id + i] > need_sample_max[local_id]))
//       need_sample_max[local_id] = need_sample_max[local_id + i];
//     if(local_id < i && (need_sample_min[local_id + i] < need_sample_min[local_id]))
//       need_sample_min[local_id] = need_sample_min[local_id + i];
//     __syncthreads();
//   }

//   int check_num = 0;

//   if(x >= image_width - 1 || x <= 0 || y >= image_height - 1 || y <= 0)
//     return;

//   for(int depth_id = need_sample_min[0]; depth_id <= need_sample_max[0]; depth_id++)
//   {
//     check_num++;
//     float check_depth = 1.0f / (start_invdepth + depth_id * inv_depth_step);
//     float2 check_pixel = camera_devptr->world2cam(cur_to_ref * (feature_dir * check_depth));
//     int2 check_int = make_int2((int)check_pixel.x, (int)check_pixel.y);
//     if( in_range(check_int, 1, 1, image_width - 1, image_height - 1) )
//     {
//       float cost(0.0f);
//       for(int shift_j = 0; shift_j < 3; shift_j ++)
//       {
//         for(int shift_i = 0; shift_i < 3; shift_i ++)
//           cost += fabs( featue_patch[shift_i][shift_j] - ref_dev->atXY(check_int.x + shift_i - 1, check_int.y + shift_j - 1));
//       }
//       cost_per_frame[pixel_id][frame_id] = cost;
//       get_value[pixel_id][frame_id] = 1.0f;
//     }
//     else
//     {
//       cost_per_frame[pixel_id][frame_id] = 0.0f;
//       get_value[pixel_id][frame_id] = 0.0f;
//     }
  
//     __syncthreads();
  
//     for(int r = 8; r > 0; r /= 2)
//     {
//       if(frame_id + r < frame_num && frame_id < r)
//       {
//         cost_per_frame[pixel_id][frame_id] += cost_per_frame[pixel_id][frame_id + r];
//         get_value[pixel_id][frame_id] += get_value[pixel_id][frame_id + r];
//       }
//       __syncthreads();
//     }
  
//     if(frame_id == 0)
//     {
//       if(get_value[pixel_id][0] != 0.0f)
//       {
//         float cost = cost_per_frame[pixel_id][0] / get_value[pixel_id][0];
//         for(int feature_index = 0; feature_index < neighbor_num; feature_index ++)
//           cost += ( depth_id - neighbor_depth[feature_index] ) * ( depth_id - neighbor_depth[feature_index] ) 
//                   * neighbor_weight[feature_index] / norm_weight;
//         (image_cost_devptr->atXY(x, y)).set_cost(depth_id, cost);
//       }
//     }
//   }
// }

__global__ void feature_cost_aggregate( MatchParameter *match_parameter_devptr,
                                        DeviceImage<int> *feature_age_devptr,
                                        DeviceImage<PIXEL_COST> *feature_cost_devptr)
{
  const int depth_id = threadIdx.x;
  const int frame_id = threadIdx.y;
  const int image_width = match_parameter_devptr->image_width;
  const int image_height = match_parameter_devptr->image_height;
  const int keypoint_step = match_parameter_devptr->keypoint_step;
  const PinholeCamera* camera_devptr = match_parameter_devptr->camera_devptr;
  const float start_invdepth = 1.0f / match_parameter_devptr->max_match_dist;
  const float inv_depth_step = match_parameter_devptr->inv_depth_step;
  const int feature_x = blockIdx.x * keypoint_step;
  const int feature_y = blockIdx.y * keypoint_step;

  if(feature_x >= image_width - 1 || feature_x <= 1 || feature_y >= image_height - 1 || feature_y <= 1)
    return;

  float feature_age = feature_age_devptr->atXY(blockIdx.x, blockIdx.y);
  feature_age = feature_age > 59 ? 59 : feature_age;
  feature_age = feature_age < 10 ? 10 : feature_age;
  const int check_id = (int)(feature_age / (float)blockDim.y * (float)frame_id);
  // const int check_id = 3;

  const DeviceImage<float> *ref_dev = match_parameter_devptr->keyframe_devptr[check_id];
  const SE3<float> cur_to_ref = match_parameter_devptr->keyframe_worldpose[check_id] * (match_parameter_devptr->current_worldpose).inv();
  float3 feature_dir = normalize(camera_devptr->cam2world(make_float2(feature_x, feature_y)));
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
  float2 check_pixel = camera_devptr->world2cam(cur_to_ref * (feature_dir * check_depth));
  int2 check_int = make_int2((int)check_pixel.x, (int)check_pixel.y);
  if( in_range(check_int, 1, 1, image_width - 1, image_height - 1) )
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



// __global__ void initial_similarity( MatchParameter *match_parameter_devptr,
//                                     DeviceImage<float4> *similarity_devptr)
// {
//   const int feature_width = similarity_devptr->width;
//   const int feature_height = similarity_devptr->height;
//   const int keypoint_step = match_parameter_devptr->keypoint_step;
//   const int image_width = match_parameter_devptr->image_width;
//   const int image_height = match_parameter_devptr->image_height;
//   const int feature_x = blockIdx.x * blockDim.x + threadIdx.x;
//   const int feature_y = blockIdx.y * blockDim.y + threadIdx.y;

//   if(feature_x >= feature_width || feature_y >= feature_height)
//     return;

//   const int image_x = feature_x * keypoint_step;
//   const int image_y = feature_y * keypoint_step;
//   int2 left = make_int2(image_x - 4, image_y);
//   int2 up = make_int2(image_x, image_y - 4);
//   int2 right = make_int2(image_x + 4, image_y);
//   int2 down = make_int2(image_x, image_y + 4);
//   left.x = left.x < 0 ? 0 : left.x;
//   up.y = up.y < 0 ? 0 : up.y;
//   right.x = right.x >= image_width ? image_width - 1 : right.x;
//   down.y = down.y >= image_height ? image_height - 1 : down.y;

//   const float this_pixel = tex2D(curr_img_tex, image_x + 0.5f, image_y + 0.5f);
//   const float left_pixel = tex2D(curr_img_tex, left.x + 0.5f, left.y + 0.5f);
//   const float up_pixel = tex2D(curr_img_tex, up.x + 0.5f, up.y + 0.5f);
//   const float right_pixel = tex2D(curr_img_tex, right.x + 0.5f, right.y + 0.5f);
//   const float down_pixel = tex2D(curr_img_tex, down.x + 0.5f, down.y + 0.5f);

//   float left_similarity = expf( - (this_pixel - left_pixel) * (this_pixel - left_pixel));
//   float right_similarity = expf( -  (this_pixel - right_pixel) * (this_pixel - right_pixel));
//   float up_similarity = expf( - (this_pixel - up_pixel) * (this_pixel - up_pixel));
//   float down_similarity = expf( - (this_pixel - down_pixel) * (this_pixel - down_pixel));
//   similarity_devptr->atXY(feature_x, feature_y) = make_float4(left_similarity, right_similarity, up_similarity, down_similarity);
//   // similarity_devptr->atXY(feature_x, feature_y) = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
// }

__global__ void LR_check( MatchParameter *match_parameter_devptr, 
                          DeviceImage<float> *depth_cost_devptr,
                          DeviceImage<int> *predice_age_devptr,
                          const SE3<float> cur_to_last, 
                          const SE3<float> cur_to_last2)
{
  const int width = match_parameter_devptr->image_width;
  const int height = match_parameter_devptr->image_height;
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= width || y >= height)
    return;

  DeviceImage<float> *depth_output_devptr = match_parameter_devptr->depth_output_devptr;
  DeviceImage<float> *depth_map_devptr = match_parameter_devptr->depth_map_devptr;
  DeviceImage<float> *previous_depth_map_devptr = match_parameter_devptr->previous_depth_map_devptr;
  DeviceImage<float> *previous2_depth_map_devptr = match_parameter_devptr->previous2_depth_map_devptr;

  float depth = depth_map_devptr->atXY(x, y);
  float inv_depth = 1.0f / depth;
  float cost = depth_cost_devptr->atXY(x, y);

  if(cost > 5.0f)
    depth_output_devptr->atXY(x,y) = -1.0f;

  const PinholeCamera *camera_devptr = match_parameter_devptr->camera_devptr;
  float3 feature_dir = normalize(camera_devptr->cam2world(make_float2(x, y)));
  float2 check_pixel1 = camera_devptr->world2cam(cur_to_last * (feature_dir * depth));
  float2 check_pixel2 = camera_devptr->world2cam(cur_to_last2 * (feature_dir * depth));
  int2 check_1 = make_int2(check_pixel1.x, check_pixel1.y);
  int2 check_2 = make_int2(check_pixel2.x, check_pixel2.y);

  if(!in_range(check_1, 0, 0, width, height))
  {
    depth_output_devptr->atXY(x, y) = -1.0f;
    return;
  }
  if(!in_range(check_2, 0, 0, width, height))
  {
    depth_output_devptr->atXY(x, y) = -1.0f;
    return;
  }

  float depth1 = previous_depth_map_devptr->atXY(check_1.x , check_1.y);
  float depth2 = previous2_depth_map_devptr->atXY(check_2.x , check_2.y);
  float inv_depth1 = 1.0f / depth1;
  float inv_depth2 = 1.0f / depth2;

  float max_inverror = 0.0314; //one desparity error 
  float max_error = 0.5; //one desparity error 

  bool consist1 = (fabs(inv_depth - inv_depth1) < max_inverror ) || (fabs(depth - depth1) < max_error);
  bool consist2 = (fabs(inv_depth - inv_depth2) < max_inverror ) || (fabs(depth - depth2) < max_error);

  if( consist1 && consist2 )
  {
    depth_output_devptr->atXY(x, y) = depth;
    return;
  }
  else
  {
    depth_output_devptr->atXY(x, y) = -1.0f;
    predice_age_devptr->atXY(x/4, y/4) = 9;
  }
}

}//namespace
