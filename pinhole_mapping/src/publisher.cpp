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

#include <rmd/publisher.h>

#include <rmd/seed_matrix.cuh>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

#include <Eigen/Eigen>

rmd::Publisher::Publisher(ros::NodeHandle &nh,
                          std::shared_ptr<rmd::Depthmap> depthmap,
                          bool publish_pc)
    : nh_(nh), pc_(new PointCloud), need_pointcloud(publish_pc)
{
  // for save image
  std::cout << " initial the publisher ! " << std::endl;

  depthmap_ = depthmap;
  image_transport::ImageTransport it(nh_);

  pub_depth = nh_.advertise<sensor_msgs::Image>("depth", 1000);
  pub_coldepth = nh_.advertise<sensor_msgs::Image>("col_depth", 1000);
  pub_ref = nh_.advertise<sensor_msgs::Image>("reference", 1000);
  pub_pc_ = nh_.advertise<PointCloud>("pointcloud", 1);
}

void rmd::Publisher::get_data()
{
  cv_depth = depthmap_->getDepthmap();
  cv_reference = depthmap_->getReferenceImage();
  T_world_ref = depthmap_->getT_world_ref();
}

void rmd::Publisher::publishDepthmap(ros::Time msg_time)
{
  //pub depth
  {
    cv_bridge::CvImage bridge_depth;
    bridge_depth.header.frame_id = "depthmap";
    bridge_depth.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    bridge_depth.image = cv_depth;
    bridge_depth.header.stamp = msg_time;
    pub_depth.publish(bridge_depth.toImageMsg());
  }

  //pub_color_depth
  {
    //color code the map
    double min;
    double max;
    cv::minMaxIdx(cv_depth, &min, &max);
    cv::Mat adjMap;
    std::cout << "min max of the depth: " << min << " , " << max << std::endl;
    min = 0.5;
    max = 10;
    cv_depth.convertTo(adjMap, CV_8UC1, 255 / (max - min), -min);
    cv::Mat falseColorsMap;
    cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_RAINBOW);

    cv::Mat mask;
    cv::inRange(falseColorsMap, cv::Scalar(0, 0, 255), cv::Scalar(0, 0, 255), mask);
    cv::Mat black_image = cv::Mat::zeros(falseColorsMap.size(), CV_8UC3);
    black_image.copyTo(falseColorsMap, mask);

    cv_bridge::CvImage bridge_colordepth;
    bridge_colordepth.header.frame_id = "depthmap";
    bridge_colordepth.encoding = sensor_msgs::image_encodings::BGR8;
    bridge_colordepth.image = falseColorsMap;
    bridge_colordepth.header.stamp = msg_time;
    pub_coldepth.publish(bridge_colordepth.toImageMsg());
  }

  //pub reference
  {
    cv::Mat cv_reference_mono;
    cv_reference.convertTo(cv_reference_mono, CV_8UC1, 255);

    cv_bridge::CvImage bridge_reference;
    bridge_reference.header.frame_id = "cv_reference";
    // bridge_reference.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    bridge_reference.encoding = sensor_msgs::image_encodings::MONO8;
    bridge_reference.image = cv_reference_mono;
    bridge_reference.header.stamp = msg_time;
    pub_ref.publish(bridge_reference.toImageMsg());
  }
}

void rmd::Publisher::publishPointCloud(ros::Time msg_time)
{
  {
    const float fx = depthmap_->getFx();
    const float fy = depthmap_->getFy();
    const float cx = depthmap_->getCx();
    const float cy = depthmap_->getCy();
    pc_->clear();

    for (int y = 0; y < cv_depth.rows; ++y)
    {
      for (int x = 0; x < cv_depth.cols; ++x)
      {
        float depth_value = cv_depth.at<float>(y, x);
        if (depth_value < 0)
          continue;
        const float3 f = normalize(make_float3((x - cx) / fx, (y - cy) / fy, 1.0f));
        const float3 xyz = T_world_ref * (f * depth_value);

        PointType p;
        p.x = xyz.x;
        p.y = xyz.y;
        p.z = xyz.z;
        const float intensity = cv_reference.at<float>(y, x) * 255.0f;
        p.intensity = intensity;
        pc_->push_back(p);
      }
    }
  }
  if (!pc_->empty())
  {
    if (nh_.ok())
    {
      pcl_conversions::toPCL(msg_time, pc_->header.stamp);
      pc_->header.frame_id = "/world";
      pub_pc_.publish(pc_);
      std::cout << "INFO: publishing pointcloud, " << pc_->size() << " points" << std::endl;
    }
  }
}

void rmd::Publisher::publishDepthmapAndPointCloud(ros::Time msg_time)
{
  get_data();
  publishDepthmap(msg_time);
  if(need_pointcloud)
    publishPointCloud(msg_time);
}
