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
#include <vector>
#include <ctime>
#include <iostream>
#include <rmd/device_image.cuh>
#include <rmd/camera_model/pinhole_camera.cuh>
#include <rmd/se3.cuh>
#include <rmd/match_parameter.cuh>
#include <rmd/pixel_cost.cuh>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

#define KEYFRAME_NUM 60

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
      const size_t &height,
      const PinholeCamera &cam);
  ~SeedMatrix();

  void set_remap(cv::Mat _remap_1, cv::Mat _remap_2);

  bool input_raw(
      cv::Mat raw_mat,
      const SE3<float> T_curr_world
    );
  bool add_frames(
      cv::cuda::GpuMat &input_image,
      const SE3<float> T_curr_world
    );

  void get_result(cv::Mat &depth, cv::Mat &reference);

private:
  bool is_keyframe(const rmd::SE3<float> &T_curr_world);
  void download_output();

  size_t width;
  size_t height;

  float max_sample_dist;
  float min_sample_dist;
  MatchParameter match_parameter;
  
  DeviceImage<float> ref_img, curr_img;
  DeviceImage<float> previous2_depthmap;
  DeviceImage<float> previous_depthmap;
  DeviceImage<float> depth_map;
  DeviceImage<float> depth_output;

  DeviceImage<int> prefeature_age;
  DeviceImage<int> curfeature_age;
  DeviceImage<float4> pre_fuse;
  DeviceImage<float4> cur_fuse;

  //device image list
  int frame_index;
  SE3<float> first_frame_pose;
  const int keyframe_list_length;
  std::deque< frame_element > keyframe_list;

  //camera model
  PinholeCamera camera;
  SE3<float> T_world_ref;

  //used for gpu remap
  cv::cuda::GpuMat remap_1, remap_2;
  cv::cuda::GpuMat input_image;
  cv::cuda::GpuMat input_float;
  cv::cuda::GpuMat undistorted_image;

  //result
  cv::Mat cv_output;
  cv::Mat cv_reference;
};

} // rmd namespace

#endif // SEED_MATRIX_CUH
