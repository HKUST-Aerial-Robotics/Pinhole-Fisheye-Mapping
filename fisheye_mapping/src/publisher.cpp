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
                          std::shared_ptr<rmd::Depthmap> depthmap)
  : nh_(nh)
  , pc_(new PointCloud)
{
  // for save image
  save_index = 0;
  std::cout << " initial the publisher ! " << std::endl;
  save_path = std::string("/home/wang/Desktop/test/pure_sgm/");

  depthmap_ = depthmap;
  colored_.create(depthmap->getHeight(), depthmap_->getWidth(), CV_8UC3);

  width = depthmap_->getWidth();
  height = depthmap->getHeight();

  image_transport::ImageTransport it(nh_);
  depthmap_publisher_ = it.advertise("depth",       10);
  colored_depthmap_publisher_ = it.advertise("colored_depthmap", 10);
  reference_publisher_ = it.advertise("reference", 10);
  pub_pc_ = nh_.advertise<PointCloud>("pointcloud", 1);

  nh_.getParam("undistort_the_depth", pub_UndistortedDepth);
  nh_.getParam("publish_Cloud", pub_Cloud);
}

void rmd::Publisher::publishUndistortedDepthmap(ros::Time msg_time)
{
  cv::Mat undistorted = cv::Mat_<float>(720, 720);
  depthmap_->getUndistortedDepthmap(reinterpret_cast<float*>(undistorted.data));

  cv_bridge::CvImage undis_depth;
  undis_depth.header.frame_id = "depthmap";
  undis_depth.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
  undis_depth.image = undistorted;
  undis_depth.header.stamp = msg_time;
  depthmap_publisher_.publish(undis_depth.toImageMsg());

  cv_bridge::CvImage colored_depth;
  double min;
  double max;
  cv::minMaxIdx(undistorted, &min, &max);
  printf("the max depth %f, min depth %f.\n", max, min);
  min = 2; max = 10;
  cv::Mat adjMap;
  undistorted.convertTo(adjMap, CV_8UC1, 255 / (max-min), -min);
  cv::Mat falseColorsMap;
  cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_RAINBOW);

  cv::Mat mask;
  cv::inRange(falseColorsMap, cv::Scalar(0, 0, 255), cv::Scalar(0, 0, 255), mask);
  cv::Mat black_image = cv::Mat::zeros(falseColorsMap.size(), CV_8UC3);
  black_image.copyTo(falseColorsMap, mask);
  
  colored_depth.header.frame_id = "coloreddepth";
  colored_depth.encoding = sensor_msgs::image_encodings::BGR8;
  colored_depth.image = falseColorsMap;
  colored_depth.header.stamp = msg_time;
  colored_depthmap_publisher_.publish(colored_depth.toImageMsg());

  cv::Mat reference = cv::Mat_<float>(720, 720);
  depthmap_->getReferenceImage(reinterpret_cast<float*>(reference.data));
  cv::Mat reference_uchar;
  reference.convertTo(reference_uchar, CV_8UC1, 255);
  cv_bridge::CvImage reference_cv;
  reference_cv.header.frame_id = "reference";
  reference_cv.encoding = sensor_msgs::image_encodings::MONO8;
  reference_cv.image = reference_uchar;
  reference_cv.header.stamp = msg_time;
  reference_publisher_.publish(reference_cv.toImageMsg());
}

void rmd::Publisher::publishDepthmap(ros::Time msg_time)
{
  cv::Mat depthmap_mat;
  cv_bridge::CvImage cv_image, cv_image_colored;
  cv_image.header.frame_id = "depthmap";
  cv_image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
  depthmap_mat = depthmap_->getDepthmap();
  cv_image.image = depthmap_mat;

  //color code the map
  double min;
  double max;
  cv::minMaxIdx(depthmap_mat, &min, &max);
  cv::Mat adjMap;
  std::cout << "min max of the depth: " << min << " , " << max << std::endl;
  min = 1; max = 20;
  // min = 0; max = 65;
  depthmap_mat.convertTo(adjMap, CV_8UC1, 255 / (max-min), -min);
  cv::Mat falseColorsMap;
  cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_RAINBOW);
  cv_image_colored.header.frame_id = "coloreddepth";
  cv_image_colored.encoding = sensor_msgs::image_encodings::BGR8;
  cv_image_colored.image = falseColorsMap;

  if(nh_.ok())
  {
    cv_image.header.stamp = msg_time;
    cv_image_colored.header.stamp = msg_time;
    depthmap_publisher_.publish(cv_image.toImageMsg());
    colored_depthmap_publisher_.publish(cv_image_colored.toImageMsg());
  }
}


void rmd::Publisher::publishPointCloud(ros::Time msg_time)
{
  std::clock_t start = std::clock();
  float* cloud_ptr;
  cloud_ptr = (float* )malloc(720 * 720 * 4 * sizeof(float));
  {
    depthmap_->get_pointcloud(cloud_ptr);

    pc_->clear();

    for(int y=0; y<720; ++y)
    {
      for(int x=0; x<720; ++x)
      {
        int index = ( y * 720 + x ) * 4;
        float certainity = cloud_ptr[index + 3];

        if(certainity > 0.0f)
        {
          PointType p;
          p.x = cloud_ptr[index];
          p.y = cloud_ptr[index + 1];
          p.z = cloud_ptr[index + 2];
          p.intensity = certainity*4.0;
          pc_->push_back(p);
        }
      }
    }
  }
  if (!pc_->empty())
  {
    if(nh_.ok())
    {
      pc_->header.frame_id = "world";
      pcl_conversions::toPCL(msg_time, pc_->header.stamp);
      pub_pc_.publish(pc_);
      std::cout << "INFO: publishing pointcloud, " << pc_->size() << " points" << std::endl;
    }
  }
  free(cloud_ptr);
  printf("publish the cloud (%d, %d) cost %f ms size %d.\n", 
    720, 720,
    ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000,
    pc_->size());
}

void rmd::Publisher::publishDepthmapAndPointCloud(ros::Time msg_time)
{
  if(pub_UndistortedDepth)
    publishUndistortedDepthmap(msg_time);
  else
    publishDepthmap(msg_time);

  if(pub_Cloud)
    publishPointCloud(msg_time);
}
