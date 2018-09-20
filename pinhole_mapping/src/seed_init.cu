// This file is part of REMODE - REgularized MOnocular Depth Estimation.
//
// Copyright (C) 2014 Matia Pizzoli <matia dot pizzoli at gmail dot com>
// Robotics and Perception Group, University of Zurich, Switzerland
// http://rpg.ifi.uzh.ch
//
// REMODE is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// REMODE is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef RMD_SEED_INIT_CU
#define RMD_SEED_INIT_CU

#include <rmd/texture_memory.cuh>
#include <rmd/pixel_cost.cuh>
#include <rmd/match_parameter.cuh>
// #include <rmd/tracked_feature.cuh>

namespace rmd
{
__global__
void seedInitKernel_NCC(  const int width, const int height, const int half_window_size,
                          DeviceImage<float> *sum_templ_ptr, DeviceImage<float> *const_templ_denom_ptr)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const float area = (2 * half_window_size + 1) * (2 * half_window_size + 1);

  if(x >= width || y >= height)
    return;

  // Compute template statistics for NCC
  float sum_templ    = 0.0f;
  float sum_templ_sq = 0.0f;

  for(int patch_y = - half_window_size; patch_y <= half_window_size; ++patch_y)
  {
    for(int patch_x = - half_window_size; patch_x <= half_window_size; ++patch_x)
    {
      const float templ = tex2D( curr_img_tex, (float)(x + patch_x + 0.5f), (float)(y + patch_y + 0.5f));
      sum_templ += templ;
      sum_templ_sq += templ * templ;
    }
  }
  sum_templ_ptr->atXY(x, y) = sum_templ;
  const_templ_denom_ptr->atXY(x, y) = (float) ( (double) area * sum_templ_sq - (double) sum_templ * sum_templ );
}

__global__ 
void sober_filter (const int width, const int height, DeviceImage<float> *image_devptr, DeviceImage<float> *sober_x_devptr, DeviceImage<float> *sober_y_devptr)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;
  __shared__ float local_image[16][16];

  if(x >= width || y >= height)
    return;

  local_image[local_x][local_y] = image_devptr->atXY(x,y);
  __syncthreads();
  
  float neighbor_pixel[3][3];
  for (int j = -1; j <=1; j++)
  {
    for (int i = -1; i <= 1; i++)
    {
      int neighbor_x = local_x + i;
      int neighbor_y = local_y + j;
      int image_x = x + i;
      int image_y = y + j;
      if(neighbor_x >= 0 && neighbor_x < blockDim.x && neighbor_y >= 0 && neighbor_y < blockDim.y && image_x < width && image_y < height)
      {
        neighbor_pixel[i+1][j+1] = local_image[neighbor_x][neighbor_y];
      }
      else
      {
        if (image_x >= width || image_x < 0)
          image_x = image_x > 0 ? (width - 1) : 0;
        if (image_y >= height || image_y < 0)
          image_y = image_y > 0 ? (height - 1) : 0;
        neighbor_pixel[i+1][j+1] = image_devptr->atXY(image_x, image_y);
      }
    }
  }

  sober_x_devptr->atXY(x,y) = (neighbor_pixel[0][2] - neighbor_pixel[0][0])
                            + 2.0f * (neighbor_pixel[1][2] - neighbor_pixel[1][0])
                            + (neighbor_pixel[2][2] - neighbor_pixel[2][0]);

  sober_y_devptr->atXY(x,y) = (neighbor_pixel[2][0] - neighbor_pixel[0][0])
                            + 2.0f * (neighbor_pixel[2][1] - neighbor_pixel[0][1])
                            + (neighbor_pixel[2][2] - neighbor_pixel[0][2]);
}

__global__ void gradient_cal( MatchParameter *match_parameter_devptr,
                              DeviceImage<float> *gradient_devptr,
                              DeviceImage<float> *gradient_theta_devptr)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;
  const int image_width = match_parameter_devptr->image_width;
  const int image_height = match_parameter_devptr->image_height;

  __shared__ float local_image[16][16];

  if(x >= image_width || y >= image_height)
    return;

  local_image[local_x][local_y] = tex2D(curr_img_tex, x + 0.5f, y + 0.5f);
  __syncthreads();
  
  float neighbor_pixel[3][3];
  for (int j = -1; j <=1; j++)
  {
    for (int i = -1; i <= 1; i++)
    {
      int neighbor_x = local_x + i;
      int neighbor_y = local_y + j;
      int image_x = x + i;
      int image_y = y + j;
      if( neighbor_x >= 0 && neighbor_x < blockDim.x && neighbor_y >= 0 &&
          neighbor_y < blockDim.y && image_x < image_width && image_y < image_height)
      {
        neighbor_pixel[i+1][j+1] = local_image[neighbor_x][neighbor_y];
      }
      else
      {
        if (image_x >= image_width || image_x < 0)
          image_x = image_x > 0 ? (image_width - 1) : 0;
        if (image_y >= image_height || image_y < 0)
          image_y = image_y > 0 ? (image_height - 1) : 0;
        neighbor_pixel[i+1][j+1] = tex2D(curr_img_tex, image_x + 0.5f, image_y + 0.5f);
      }
    }
  }
  float sober_x = (neighbor_pixel[0][2] - neighbor_pixel[0][0])
                  + 2.0f * (neighbor_pixel[1][2] - neighbor_pixel[1][0])
                  + (neighbor_pixel[2][2] - neighbor_pixel[2][0]);
  float sober_y = (neighbor_pixel[2][0] - neighbor_pixel[0][0])
                  + 2.0f * (neighbor_pixel[2][1] - neighbor_pixel[0][1])
                  + (neighbor_pixel[2][2] - neighbor_pixel[0][2]);
  float gradient = sqrtf(sober_x * sober_x + sober_y * sober_y);
  float theta = atan2f(fabs(sober_y), fabs(sober_x));
  gradient_devptr->atXY(x,y) = gradient;
  gradient_theta_devptr->atXY(x,y) = theta;
}

} // rmd namespace

#endif
