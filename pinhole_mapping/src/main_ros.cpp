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
#include <ros/ros.h>
#include <rmd/check_cuda_device.cuh>
#include <rmd/depthmap_node.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <geometry_msgs/PoseStamped.h>

typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, geometry_msgs::PoseStamped> exact_policy_nocloud;

int main(int argc, char **argv)
{
  if(!rmd::checkCudaDevice(argc, argv))
    return EXIT_FAILURE;

  ros::init(argc, argv, "rpg_open_remode");
  ros::NodeHandle nh("~");
  rmd::DepthmapNode dm_node(nh);
  if(!dm_node.init())
  {
    ROS_ERROR("could not initialize DepthmapNode. Shutting down node...");
    return EXIT_FAILURE;
  }
  
  message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, "image", 1000);
  message_filters::Subscriber<geometry_msgs::PoseStamped> pose_sub(nh, "posestamped", 1000);
  message_filters::Synchronizer<exact_policy_nocloud> sync(exact_policy_nocloud(1000), image_sub, pose_sub);

  sync.registerCallback(boost::bind(&rmd::DepthmapNode::Msg_Callback, &dm_node, _1, _2));

  while(ros::ok())
  {
    ros::spinOnce();
  }

  return EXIT_SUCCESS;
}
