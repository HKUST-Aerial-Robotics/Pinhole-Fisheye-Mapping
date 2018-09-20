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
#include <rmd/seed_matrix.cuh>

#include <rmd/texture_memory.cuh>
#include <rmd/helper_vector_types.cuh>
#include <iostream>

#include "seed_init.cu"
#include "cuda_functions.cu"
#include "depth_fusion.cu"



rmd::SeedMatrix::SeedMatrix(
    const size_t &_width,
    const size_t &_height,
    const PinholeCamera &cam)
  : width(_width)
  , height(_height)
  , camera(cam)
  , ref_img(_width, _height)
  , curr_img(_width, _height)
  , depth_output(_width, _height)
  , depth_map(_width, _height)
  , previous_depthmap(_width, _height)
  , previous2_depthmap(_width, _height)
  , prefeature_age((_width+3)/4, (_height+3)/4)
  , curfeature_age((_width+3)/4, (_height+3)/4)
  , pre_fuse(_width, _height)
  , cur_fuse(_width, _height)
  , max_sample_dist(20.0f)
  , min_sample_dist(0.5f)
  , match_parameter(_width, _height)
  , keyframe_list_length(KEYFRAME_NUM)
  , frame_index(0)
{
  cv_output.create(height, width, CV_32FC1);

  camera.setDevData();

  match_parameter.set_image_devptr(ref_img.dev_ptr, curr_img.dev_ptr, depth_map.dev_ptr);
  match_parameter.set_camera_devptr(camera.dev_ptr);
  match_parameter.current_image_hostptr = &curr_img;

  match_parameter.depth_output_devptr = depth_output.dev_ptr;
  match_parameter.previous_depth_map_devptr = previous_depthmap.dev_ptr;
  match_parameter.previous2_depth_map_devptr = previous2_depthmap.dev_ptr;

  match_parameter.previous2_depth_map_hostptr = &previous2_depthmap;
  match_parameter.previous_depth_map_hostptr = &previous_depthmap;
  match_parameter.depth_map_hostptr = &depth_map;

  prefeature_age.zero();
  match_parameter.prefeature_age_hostptr = &prefeature_age;
  match_parameter.prefeature_age_devptr = prefeature_age.dev_ptr;
  match_parameter.curfeature_age_hostptr = &curfeature_age;
  match_parameter.curfeature_age_devptr = curfeature_age.dev_ptr;

  pre_fuse.zero();
  cur_fuse.zero();
  match_parameter.pre_fuse_hostptr = &pre_fuse;
  match_parameter.pre_fuse_devptr  = pre_fuse.dev_ptr;
  match_parameter.cur_fuse_hostptr = &cur_fuse;
  match_parameter.cur_fuse_devptr  = cur_fuse.dev_ptr;
  match_parameter.setDevData();

  depth_fuse_initialize(match_parameter);
}

rmd::SeedMatrix::~SeedMatrix()
{
  for(int i = 0; i < keyframe_list_length; i++)
  {
    delete keyframe_list[i].frame_hostptr;
    keyframe_list[i].frame_hostptr = NULL;
  }
}

void rmd::SeedMatrix::set_remap(cv::Mat _remap_1, cv::Mat _remap_2)
{
  remap_1.upload(_remap_1);
  remap_2.upload(_remap_2);
  printf("has success set cuda remap.\n");
}

bool rmd::SeedMatrix::input_raw(cv::Mat raw_mat, const SE3<float> T_curr_world)
{
  std::clock_t start = std::clock();  
  input_image.upload(raw_mat);
  cv::cuda::remap(input_image, undistorted_image, remap_1, remap_2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
  undistorted_image.convertTo(input_float,CV_32F,1.0/255.0f);
  printf("cuda prepare the image cost %f ms \n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);
  return add_frames(input_float, T_curr_world);
}

bool rmd::SeedMatrix::is_keyframe(const rmd::SE3<float> &T_curr_world)
{
  return true;
  
  SE3<float> curr_pose = T_curr_world.inv();
  float3 optical1 = make_float3(curr_pose.data(0,2), curr_pose.data(1,2), curr_pose.data(2,2));

  if(keyframe_list.size() == 0)
    return true;
  else
  {
    SE3<float> last_world_pose = keyframe_list[0].world_pose;
    SE3<float> last_pose = last_world_pose.inv();
    //if translate is too big
    float3 trans = curr_pose.getTranslation() - last_pose.getTranslation();
    float trans_length = length(trans);
    if(trans_length > 0.01f)
    {
      return true;
    }
    //if angle is too big
    float3 optical2 = make_float3(last_pose.data(0,2), last_pose.data(1,2), last_pose.data(2,2));
    float cos_theta = dot(optical1, optical2);
    if(cos_theta < cosf(3.0f / 180.0f * 3.141593))
    {
      return true;
    }
  }
  
  return false;
}

bool rmd::SeedMatrix::add_frames(
    cv::cuda::GpuMat &input_image,
    const SE3<float> T_curr_world)
{
  std::clock_t start = std::clock();
  frame_index++;
  const SE3<float> current_pose = T_curr_world.inv();
  curr_img.setDevData(input_image);
  printf("new frame %d, has keyframe %d!\n", frame_index, keyframe_list.size());

  bool add_keyframe = is_keyframe(T_curr_world);
  if(add_keyframe)
  {
    frame_element new_element;
    new_element.frame_hostptr = new DeviceImage<float>(width, height);
    new_element.world_pose = T_curr_world;
    *new_element.frame_hostptr = curr_img;
    keyframe_list.push_front(new_element);
    if(keyframe_list.size() > keyframe_list_length)
    {
      frame_element to_delete_element = keyframe_list.back();
      delete to_delete_element.frame_hostptr;
      keyframe_list.pop_back();
    }

    for(int i  = 0; i < keyframe_list.size(); i++)
    {
      match_parameter.keyframe_devptr[i] = keyframe_list[i].frame_hostptr->dev_ptr;
      SE3<float> keypose = (keyframe_list[i].world_pose).inv();
      float3 keyframe_dir = (T_curr_world * keypose).getTranslation();
      float2 keyframe_dir_xy = make_float2(keyframe_dir.x , keyframe_dir.y);
      float keyframe_dir_xy_length = length(keyframe_dir_xy);
      match_parameter.keyframe_worldpose[i] = keyframe_list[i].world_pose;
      match_parameter.keyframe_theta[i] = fabs(acosf( keyframe_dir_xy.x / keyframe_dir_xy_length));
    }
  }

  if(keyframe_list.size() < 10)
    return false;

  //set information
  match_parameter.previous_worldpose = match_parameter.current_worldpose;
  match_parameter.current_worldpose = T_curr_world;
  match_parameter.frame_nums++;

  rmd::bindTexture(curr_img_tex, curr_img, cudaFilterModeLinear);
  printf("till prepare the image cost %f ms.\n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);

  match_parameter.current_to_key = first_frame_pose * T_curr_world.inv();
  match_parameter.setDevData();
  printf("till prepare the keyframe cost %f ms \n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);

  //now process
  depth_map.zero();
  depth_output.zero();
  
  //match the frame to the key frame
  match_features(match_parameter, add_keyframe);

  depth_fuse(match_parameter);

  cudaUnbindTexture(curr_img_tex);
  printf("cuda total cost %f ms \n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);

  download_output();  
  return true;
}

void rmd::SeedMatrix::download_output()
{
  std::clock_t start = std::clock();
  depth_output.getDevData(reinterpret_cast<float*>(cv_output.data));
  input_float.download(cv_reference);
  printf("download depth map cost %f ms \n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);  
}

void rmd::SeedMatrix::get_result(cv::Mat &depth, cv::Mat &reference)
{
  depth = cv_output.clone();
  reference = cv_reference.clone();
}
