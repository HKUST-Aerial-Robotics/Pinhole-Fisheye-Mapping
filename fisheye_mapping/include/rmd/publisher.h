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

#ifndef RMD_PUBLISHER_H
#define RMD_PUBLISHER_H

#include <rmd/depthmap.h>
#include <Eigen/Eigen>
#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <string>
#include <sstream>
#include <ctime>
#include <pcl_ros/point_cloud.h>
#include "camera_model/camera_models/CameraFactory.h"
#include "camera_model/camera_models/PolyFisheyeCamera.h"

using namespace camera_model;

namespace rmd
{

class Publisher
{

  typedef pcl::PointXYZI PointType;
  typedef pcl::PointCloud<PointType> PointCloud;

public:

  Publisher(ros::NodeHandle &nh,
            std::shared_ptr<rmd::Depthmap> depthmap);

  void publishDepthmap(ros::Time msg_time);

  void publishUndistortedDepthmap(ros::Time msg_time);

  void publishDepthmapAndPointCloud(ros::Time msg_time);

  void publishPointCloud(ros::Time msg_time);

private:

  ros::NodeHandle &nh_;
  std::shared_ptr<rmd::Depthmap> depthmap_;

  //for save image
  int save_index;
  std::string save_path;

  int width;
  int height;

  image_transport::Publisher depthmap_publisher_;
  image_transport::Publisher reference_publisher_;
  image_transport::Publisher colored_depthmap_publisher_;

  bool pub_UndistortedDepth;
  bool pub_Cloud;

  PointCloud::Ptr pc_;
  ros::Publisher pub_pc_;

  cv::Mat colored_;
};

} // rmd namespace

#endif // PUBLISHER_H
