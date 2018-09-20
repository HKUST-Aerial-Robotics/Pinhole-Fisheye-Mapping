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
                        float fx,
                        float cx,
                        float fy,
                        float cy,
                        cv::Mat remap_1,
                        cv::Mat remap_2)
  : width_(width)
  , height_(height)
  , seeds_(width, height, rmd::PinholeCamera(fx, fy, cx, cy))
  , fx_(fx)
  , fy_(fy)
  , cx_(cx)
  , cy_(cy)
{
  seeds_.set_remap(remap_1, remap_2);

  printf("initial the seed (%d x %d) fx: %f, cx: %f, fy: %f, cy: %f.\n", width, height, fx, fy, cx, fy);
}


bool rmd::Depthmap::add_frames( const cv::Mat &img_curr,
                                const SE3<float> &T_curr_world)
{
  bool has_result;
  has_result = seeds_.input_raw(img_curr, T_curr_world);

  if(has_result)
  {
    seeds_.get_result(depth_out, reference_out);
    printf("get depth %d x %d \n.", depth_out.cols, depth_out.rows);
    printf("get reference %d x %d \n.", reference_out.cols, reference_out.rows);
    T_world_ref = T_curr_world.inv();
  }

  return has_result;
}

const cv::Mat_<float> rmd::Depthmap::getDepthmap() const
{
  return depth_out;
}

const cv::Mat rmd::Depthmap::getReferenceImage() const
{
  return reference_out;
}