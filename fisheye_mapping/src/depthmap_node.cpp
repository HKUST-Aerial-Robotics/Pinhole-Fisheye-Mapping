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

#include <rmd/depthmap_node.h>

#include <rmd/se3.cuh>

#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <vikit/params_helper.h>
#include <string>
#include <future>

using namespace std;

rmd::DepthmapNode::DepthmapNode(ros::NodeHandle &nh)
  : nh_(nh)
  , num_msgs_(0)
  , default_max_distence(20.0f)
  , default_min_distence(0.1f){}

bool rmd::DepthmapNode::init()
{
  int cam_width;
  int cam_height;
  string camera_calib_path;

  nh_.getParam("default_max_distence", default_max_distence);
  nh_.getParam("default_min_distence", default_min_distence);
  nh_.getParam("cam_width", cam_width);
  nh_.getParam("cam_height", cam_height);
  nh_.getParam("calib_path", camera_calib_path);

  depthmap_ = std::make_shared<rmd::Depthmap>(cam_width,
                                              cam_height,
                                              camera_calib_path);

  publisher_.reset(new rmd::Publisher(nh_, depthmap_));

  return true;
}

void rmd::DepthmapNode::denoiseAndPublishResults()
{
  depthmap_->downloadDepthmap();
  std::async(std::launch::async,
             &rmd::Publisher::publishDepthmapAndPointCloud,
             *publisher_,
             curret_msg_time);
}

void rmd::DepthmapNode::Msg_Callback(
    const sensor_msgs::ImageConstPtr &image_input,
    const geometry_msgs::PoseStampedConstPtr &pose_input)
{
  num_msgs_ += 1;
  curret_msg_time = image_input->header.stamp;
  if(!depthmap_)
  {
    ROS_ERROR("depthmap not initialized. Call the DepthmapNode::init() method");
    return;
  }
  cv::Mat img_8uC1, img_8uC1_half;
  try
  {
    cv_bridge::CvImageConstPtr cv_img_ptr =
        cv_bridge::toCvShare(image_input, sensor_msgs::image_encodings::MONO8);
    img_8uC1 = cv_img_ptr->image;
    // cv::resize(img_8uC1, img_8uC1_half, cv::Size(), 0.5, 0.5);
    // cv::equalizeHist(img_8uC1_half, img_8uC1_half);
    // img_8uC1 = img_8uC1_half;
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
  }
  rmd::SE3<float> T_world_curr(
        pose_input->pose.orientation.w,
        pose_input->pose.orientation.x,
        pose_input->pose.orientation.y,
        pose_input->pose.orientation.z,
        pose_input->pose.position.x,
        pose_input->pose.position.y,
        pose_input->pose.position.z);

  bool has_result;
  has_result = depthmap_->add_frames(img_8uC1, T_world_curr.inv());
  if(has_result)
    denoiseAndPublishResults();
}