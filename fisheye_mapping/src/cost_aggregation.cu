#ifndef COST_AGGREGATIONKERNEL_CU
#define COST_AGGREGATIONKERNEL_CU

#include <float.h>
#include <rmd/se3.cuh>
#include <rmd/seed_matrix.cuh>
#include <rmd/texture_memory.cuh>
#include <rmd/pixel_cost.cuh>
#include <rmd/helper_vector_types.cuh>
#include <stdio.h>

namespace rmd
{
//function declear here!
void SAD_aggregation(MatchParameter &match_parameter);

//launch as
// image_block.x = DEPTH_NUM;
// image_grid.x = image_width;
// image_grid.y = image_height;
__global__
void SAD_kernel( MatchParameter *match_parameter_devptr);



//function define here!
void SAD_aggregation(MatchParameter &match_parameter)
{
  printf("  cost aggregation!\n");
  std::clock_t start = std::clock();
  const int image_width = match_parameter.image_width;
  const int image_height = match_parameter.image_height;

  dim3 image_block;
  dim3 image_grid;
  image_block.x = 8;
  image_block.y = DEPTH_NUM/8;
  image_grid.x = image_width;
  image_grid.y = image_height;
  SAD_kernel<<<image_grid, image_block>>>(match_parameter.dev_ptr);
  cudaDeviceSynchronize();
  printf("  cost aggregation cost %f ms\n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);
}

__global__
void SAD_kernel( MatchParameter *match_parameter_devptr)
{
  const int depth_id = threadIdx.x + threadIdx.y * blockDim.x;
  const int image_width = match_parameter_devptr->image_width;
  const int image_height = match_parameter_devptr->image_height;
  const int2 this_point = make_int2(blockIdx.x, blockIdx.y);
  const float max_depth = match_parameter_devptr->max_match_dist;
  const float min_depth = match_parameter_devptr->min_match_dist;
  const int pixel_match_halfsize = match_parameter_devptr->pixel_match_halfsize;
  const int pixel_match_size = 2 * pixel_match_halfsize + 1;
  const PinholeCamera *camera_devptr = match_parameter_devptr->camera_devptr;
  const SE3<float> current_to_key = match_parameter_devptr->current_to_key;
  DeviceImage<PIXEL_COST> *cost_field_devptr = match_parameter_devptr->cost_field_devptr;

  const float this_invdepth  = (1.0f/min_depth - 1.0f/max_depth) * (depth_id + 1) / (float) DEPTH_NUM
                    + 1.0f/max_depth;
  const float this_depth = 1.0f / this_invdepth;

  __shared__ float this_patch[7][7];
  if(threadIdx.x < pixel_match_size && threadIdx.y < pixel_match_size)
    this_patch[threadIdx.x][threadIdx.y] = tex2D( curr_img_tex,
                                                  this_point.x + threadIdx.x - pixel_match_halfsize + 0.5f,
                                                  this_point.y + threadIdx.y - pixel_match_halfsize + 0.5f);
  __syncthreads();
  float3 point_dir = normalize(camera_devptr->cam2world( make_float2(this_point) ));
  float2 measure_point = camera_devptr->world2cam(current_to_key * (point_dir * this_depth) );
  int2 measure_point_int = make_int2(measure_point.x, measure_point.y);

  if(measure_point_int.x < pixel_match_halfsize || measure_point_int.y < pixel_match_halfsize
      || measure_point_int.x >= (image_width - pixel_match_halfsize) || measure_point_int.y >= (image_height - pixel_match_halfsize))
  {
    (cost_field_devptr->atXY(this_point.x, this_point.y)).add_cost(depth_id, 999.9f);
    return;
  }

  float this_cost(0.0f);
  for(int j = 0; j < pixel_match_size; j++)
  {
    for(int i = 0; i < pixel_match_size; i++)
    {
      float check_pixel = tex2D ( keyframe_tex,
                                  measure_point_int.x + i - pixel_match_halfsize + 0.5f,
                                  measure_point_int.y + j - pixel_match_halfsize + 0.5f);
      this_cost += fabs(check_pixel - this_patch[i][j]);
    }
  }
  (cost_field_devptr->atXY(this_point.x, this_point.y)).add_cost(depth_id, this_cost);
}

// __global__
// void ncc_CostAggregationKernel( const int width, const int height, const int Half_Windows_Size,
//                                 const SE3<float> T_curr_ref, const PinholeCamera *camera_dev_ptr,
//                                 DeviceImage<PIXEL_COST> *pixel_cost_ptr
//                               )
// {
//   //!!! this can be accelerated by shared memory
//   const int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
//   const int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
//   const int initial_depth_level = threadIdx.z;
//   const float Window_Area = (2 * Half_Windows_Size + 1) * (2 * Half_Windows_Size + 1);

//   //if the pixel is near the board, return
//   if(pixel_x >= width - Half_Windows_Size || pixel_y >= height - Half_Windows_Size)
//     return;
//   if(pixel_x <= Half_Windows_Size || pixel_y <= Half_Windows_Size)
//     return;

//   //how many iterater
//   const int pixel_depthnum = DEPTH_NUM;
//   const int depth_batch = blockDim.z;
//   const int iterate_times = pixel_depthnum / depth_batch;

//   // Retrieve template statistics for NCC matching;
//   const float sum_templ = tex2D(sum_templ_tex, pixel_x, pixel_y);
//   const float const_templ_denom = tex2D(const_templ_denom_tex, pixel_x, pixel_y);

//   for (int iterate = 0; iterate < iterate_times; iterate ++)
//   {
//     //this thread's depth
//     const int now_depth_level = initial_depth_level + iterate * depth_batch;
//     float depth = (pixel_cost_ptr->atXY(pixel_x, pixel_y)).get_depth(now_depth_level);

//     bool can_update     = true;
//     float sum_img       = 0.0f;
//     float sum_img_sq    = 0.0f;
//     float sum_img_templ = 0.0f;

//     //!!!this may be optimized, for the patch may still be the same patch after the transformation when the frames are near
//     for(int patch_y = - Half_Windows_Size; patch_y <= Half_Windows_Size; ++patch_y)
//     {
//       for(int patch_x = - Half_Windows_Size; patch_x <= Half_Windows_Size; ++patch_x)
//       {
//         float temp_x = pixel_x + patch_x;
//         float temp_y = pixel_y + patch_y;

//         float2 px_ref = make_float2(temp_x, temp_y);
//         float3 dir_ref = normalize(camera_dev_ptr->cam2world(px_ref));
//         float2 px_curr = camera_dev_ptr->world2cam(T_curr_ref * (dir_ref * depth));

//         //maybe break???
//         if(px_curr.x >= width - Half_Windows_Size || px_curr.y >= height - Half_Windows_Size || 
//           px_curr.x <= Half_Windows_Size || px_curr.y <= Half_Windows_Size)
//         {
//           can_update = false;
//           continue;
//         }

//         const float templ = tex2D(ref_img_tex, px_ref.x + 0.5f, px_ref.y + 0.5f);
//         const float img = tex2D(curr_img_tex, px_curr.x + 0.5f, px_curr.y + 0.5f);

//         sum_img    += img;
//         sum_img_sq += img * img;
//         sum_img_templ += img * templ;
//       }
//     }

//     if (can_update)
//     {
//       const float ncc_numerator = Window_Area * sum_img_templ - sum_img * sum_templ;
//       const float ncc_denominator = (Window_Area * sum_img_sq - sum_img * sum_img) * const_templ_denom;
//       const float ncc = ncc_numerator * rsqrtf(ncc_denominator + 10.0f);

//       if(ncc != ncc)
//       {
//         printf("sum_img: %f, sum_img_sq: %f, sum_img_templ: %f,ncc_numerator: %f, ncc_denominator: %f, ncc: %f\n",
//           sum_img, sum_img_sq, sum_img_templ, ncc_numerator, ncc_denominator, ncc);
//       }
//       //update the ncc value
//       (pixel_cost_ptr->atXY(pixel_x, pixel_y)).add_cost(now_depth_level, - 1.0 * ncc);
//     }
//   }
// }

} //rmd namedspace

#endif //COST_AGGREGATIONKERNEL_CU