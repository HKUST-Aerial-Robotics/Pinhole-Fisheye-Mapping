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

#ifndef SEED_MATRIX_CUH
#define SEED_MATRIX_CUH

#include <cuda_runtime.h>
#include <cstdlib>
#include <deque>
#include <ctime>
#include <iostream>
#include <vector>
#include <utility>      // std::pair
#include <rmd/device_image.cuh>
#include <rmd/se3.cuh>
#include <rmd/match_parameter.cuh>
// #include <rmd/feature_pixel.cuh>
#include <rmd/pixel_cost.cuh>
#include <rmd/tracked_feature.cuh>
#include <opencv2/opencv.hpp>

#define KEYFRAME_NUM 30
#define KEYSTEP 4
//every UNDISTORT_STEP a degree to sample
#define UNDISTORT_STEP 4 

namespace rmd
{

struct frame_element
{
  DeviceImage<float> *frame_hostptr;
  SE3<float> world_pose;
  frame_element()
  {
    frame_hostptr = NULL;
  }
};

class SeedMatrix
{
public:
  SeedMatrix(
      const size_t &width,
      const size_t &height);

  ~SeedMatrix();

  void set_fisheye( const std::vector<double> &a2r,
                    const std::vector<double> &r2a,
                    const FisheyeParam fisheyepara);

  bool add_frames(
      cv::Mat _add_mat,
      float *host_img_align_row_maj,
      const SE3<float> T_curr_world);

  void get_depthmap(
    float *host_denoised);
  
  void get_pointcloud(
    float *host_pointcloud);

  void get_undistorted_depthmap(
    float *host_undistorted);

  void get_ReferenceImage(
    float *host_reference);

  const DeviceImage<float> & getRefImg()const;

private:
  bool is_keyframe(const rmd::SE3<float> &T_curr_world);
  void undistorted_depthmap();
  void undistorted_reference();
  
  size_t width;
  size_t height;

  float max_sample_dist;
  float min_sample_dist;
  MatchParameter match_parameter;
  
  int half_window_size;

  cv::Mat ref_mat;
  cv::Mat cur_mat;
  DeviceImage<float> ref_img, curr_img;
  DeviceImage<float> previous2_depthmap;
  DeviceImage<float> previous_depthmap;
  DeviceImage<float> depth_map;
  DeviceImage<float> depth_output;
  DeviceImage<int> prefeature_age;
  DeviceImage<int> curfeature_age;

  //for publish pointcloud
  DeviceImage<float3> pixel_dir;
  DeviceImage<float4> pub_cloud;

  //for depth image filter
  DeviceImage<float4> predepth_fuse;
  DeviceImage<float4> curdepth_fuse;

  //for publish undistorted depthmap
  DeviceImage<int2> undistorted_map;
  DeviceImage<float> undistorted_ref;
  DeviceImage<float> undistorted_depth;

  //for fisheye model
  bool has_fisheye_table;
  cudaArray* angleToR_array;
  cudaArray* rToAngle_array;
  DeviceImage<uchar> fisheye_mask;

  //device image list
  int frame_index;
  SE3<float> first_frame_pose;
  const int keyframe_list_length;
  std::deque< frame_element > keyframe_list;

  SE3<float> T_world_ref;
  float dist_from_ref;

  // kernel config
  dim3 dim_block_image;
  dim3 dim_grid_image;
};

} // rmd namespace

#endif // SEED_MATRIX_CUH
