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

#ifndef RMD_DEPTHMAP_H
#define RMD_DEPTHMAP_H

#include <memory>
#include <Eigen/Eigen>
#include <opencv2/imgproc/imgproc.hpp>
#include <rmd/seed_matrix.cuh>
#include <rmd/se3.cuh>
#include <rmd/cost_method.cuh>
#include <mutex>
#include <iostream>
#include <rmd/camera_model/fisheye_param.cuh>
#include <vector>
#include <rmd/tracked_feature.cuh>
#include "camera_model/camera_models/CameraFactory.h"
#include "camera_model/camera_models/PolyFisheyeCamera.h"

using namespace camera_model;
using namespace std;

namespace rmd
{

class Depthmap
{
public:
  Depthmap(
      size_t width,
      size_t height,
      string camera_path);
      
  bool add_frames(  const cv::Mat &img_curr,
                    const SE3<float> &T_curr_world);

  void downloadDepthmap();

  const cv::Mat_<float> getDepthmap() const;

  const cv::Mat getReferenceImage() const;
  
  // Scale depth in [0,1] and cvt to color
  // only for test and debug
  // static cv::Mat scaleMat(const cv::Mat &depthmap);

  std::mutex & getRefImgMutex()
  { return ref_img_mutex_; }

  SE3<float> getT_world_ref() const
  { return T_world_ref_; }

  float getDistFromRef() const;

  void get_pointcloud(float *cloud_ptr);
  void getUndistortedDepthmap(float *depth_ptr);
  void getReferenceImage(float *reference_ptr);

  size_t getWidth() const
  { return width_; }

  size_t getHeight() const
  { return height_; }

  PolyFisheyeCamera* get_camera_ptr()
  {
    return &fisheye_cam;
  }

private:
  void inputImage(const cv::Mat &img_8uc1);

  SeedMatrix seeds_;
  PolyFisheyeCamera fisheye_cam;

  size_t width_;
  size_t height_;

  cv::Mat undist_map1_, undist_map2_;
  cv::Mat img_undistorted_32fc1_;
  cv::Mat img_undistorted_8uc1_;
  cv::Mat ref_img_undistorted_8uc1_;
  SE3<float> T_world_ref_;
  std::mutex ref_img_mutex_;

  bool is_distorted_;

  cv::Mat output_depth_32fc1_;
};

}

#endif // RMD_DEPTHMAP_H
