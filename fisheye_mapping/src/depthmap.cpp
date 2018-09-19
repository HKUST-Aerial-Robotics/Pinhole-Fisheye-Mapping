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

#include <rmd/depthmap.h>

rmd::Depthmap::Depthmap(size_t width,
                        size_t height,
                        string camera_path)
  : width_(width)
  , height_(height)
  , is_distorted_(false)
  , seeds_(width, height)
{
  output_depth_32fc1_ = cv::Mat_<float>(height_, width_);
  img_undistorted_32fc1_.create(height_, width_, CV_32FC1);
  img_undistorted_8uc1_.create(height_, width_, CV_8UC1);
  ref_img_undistorted_8uc1_.create(height_, width_, CV_8UC1);

  //set the cameramodel
  PolyFisheyeCamera::Parameters parameters;
  parameters.readFromYamlFile(camera_path);
  fisheye_cam.setParameters(parameters);
  eigen_utils::Matrix angleToR = fisheye_cam.getMatAngleToR();
  eigen_utils::Matrix rToAngle = fisheye_cam.getMatRToAngle();

  FisheyeParam fisheye_param;
  fisheye_param.inv_k[0] = fisheye_cam.getInv_K11();
  fisheye_param.inv_k[1] = fisheye_cam.getInv_K12();
  fisheye_param.inv_k[2] = fisheye_cam.getInv_K13();
  fisheye_param.inv_k[3] = fisheye_cam.getInv_K22();
  fisheye_param.inv_k[4] = fisheye_cam.getInv_K23();
  fisheye_param.diff_r = fisheye_cam.get_diffr();
  fisheye_param.diff_angle = fisheye_cam.get_diffangle();
  fisheye_param.maxIncidentAngle = fisheye_cam.getMaxIncidentAngle();
  fisheye_param.A11 = parameters.A11();
  fisheye_param.A12 = parameters.A12();
  fisheye_param.A22 = parameters.A22();
  fisheye_param.u0 = parameters.u0();
  fisheye_param.v0 = parameters.v0();
  fisheye_param.width = fisheye_cam.imageWidth();
  fisheye_param.height = fisheye_cam.imageHeight();
  vector<double> a2r;
  vector<double> r2a;
  a2r.reserve(angleToR.rows());
  r2a.reserve(rToAngle.rows());
  printf("the table has size %dx%d\n", angleToR.rows(), angleToR.cols());
  for(int i = 0; i < angleToR.rows(); i++)
  {
    a2r.push_back(angleToR(i,1));
    r2a.push_back(rToAngle(i,1));
  }
  seeds_.set_fisheye(a2r, r2a, fisheye_param);
}

bool rmd::Depthmap::add_frames( const cv::Mat &img_curr,
                                const SE3<float> &T_curr_world)
{
  inputImage(img_curr);
  bool has_result;
  has_result = seeds_.add_frames(
    img_undistorted_32fc1_,
    reinterpret_cast<float*>(img_undistorted_32fc1_.data),
    T_curr_world);
  {
    std::lock_guard<std::mutex> lock(ref_img_mutex_);
    img_undistorted_8uc1_.copyTo(ref_img_undistorted_8uc1_);
    T_world_ref_ = T_curr_world.inv();
  }
  return has_result;
}

void rmd::Depthmap::get_pointcloud(float *cloud_ptr)
{
  seeds_.get_pointcloud(cloud_ptr);
}

void rmd::Depthmap::inputImage(const cv::Mat &img_8uc1)
{
  if(is_distorted_)
  {
    cv::remap(img_8uc1, img_undistorted_8uc1_, undist_map1_, undist_map2_, CV_INTER_LINEAR);
  }
  else
  {
    img_undistorted_8uc1_ = img_8uc1;
  }
  img_undistorted_8uc1_.convertTo(img_undistorted_32fc1_, CV_32F, 1.0f/255.0f);
  // img_undistorted_8uc1_.convertTo(img_undistorted_32fc1_, CV_32F);
}

void rmd::Depthmap::downloadDepthmap()
{
  seeds_.get_depthmap(
    reinterpret_cast<float*>(output_depth_32fc1_.data)
    );
}

void rmd::Depthmap::getUndistortedDepthmap(float *depth_ptr)
{
  seeds_.get_undistorted_depthmap(depth_ptr);
}

void rmd::Depthmap::getReferenceImage(float *reference_ptr)
{
  seeds_.get_ReferenceImage(reference_ptr);
}

const cv::Mat_<float> rmd::Depthmap::getDepthmap() const
{
  return output_depth_32fc1_;
}

const cv::Mat rmd::Depthmap::getReferenceImage() const
{
  return ref_img_undistorted_8uc1_;
}
