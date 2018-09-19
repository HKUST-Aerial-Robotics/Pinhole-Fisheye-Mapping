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

#include "cuda_functions.cu"
#include "depth_fusion.cu"


rmd::SeedMatrix::SeedMatrix(
    const size_t &_width,
    const size_t &_height)
  : width(_width)
  , height(_height)
  , fisheye_mask(_width, _height)
  , ref_img(_width, _height)
  , curr_img(_width, _height)
  , depth_output(_width, _height)
  , depth_map(_width, _height)
  , previous_depthmap(_width, _height)
  , previous2_depthmap(_width, _height)
  , prefeature_age((_width+KEYSTEP-1)/KEYSTEP, (_height+KEYSTEP-1)/KEYSTEP)
  , curfeature_age((_width+KEYSTEP-1)/KEYSTEP, (_height+KEYSTEP-1)/KEYSTEP)
  , pixel_dir(_width, _height)
  , pub_cloud(180*UNDISTORT_STEP, 180*UNDISTORT_STEP)
  , undistorted_map(180*UNDISTORT_STEP, 180*UNDISTORT_STEP)
  , undistorted_depth(180*UNDISTORT_STEP, 180*UNDISTORT_STEP)
  , undistorted_ref(180*UNDISTORT_STEP, 180*UNDISTORT_STEP)
  , predepth_fuse(180*UNDISTORT_STEP, 180*UNDISTORT_STEP)
  , curdepth_fuse(180*UNDISTORT_STEP, 180*UNDISTORT_STEP)
  , max_sample_dist(20.0f)
  , min_sample_dist(0.5f)
  , match_parameter(_width, _height, KEYSTEP)
  , keyframe_list_length(KEYFRAME_NUM)
  , frame_index(0)
  , has_fisheye_table(false)
{
  //Kernel configuration for depth estimation
  dim_block_image.x = 16;
  dim_block_image.y = 16;
  dim_grid_image.x = (_width  + dim_block_image.x - 1) / dim_block_image.x;
  dim_grid_image.y = (_height + dim_block_image.y - 1) / dim_block_image.y;

  match_parameter.set_image_devptr(ref_img.dev_ptr, curr_img.dev_ptr, depth_map.dev_ptr);
  match_parameter.set_mat_image(&ref_mat, &cur_mat);
  match_parameter.current_image_hostptr = &curr_img;

  match_parameter.depth_output_devptr = depth_output.dev_ptr;
  match_parameter.depth_output_hostptr = &depth_output;
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

  predepth_fuse.zero();
  match_parameter.predepth_fuse_devptr = predepth_fuse.dev_ptr;
  match_parameter.predepth_fuse_hostptr = &predepth_fuse;
  match_parameter.curdepth_fuse_devptr = curdepth_fuse.dev_ptr;
  match_parameter.curdepth_fuse_hostptr = &curdepth_fuse;

  match_parameter.depth_undistorted_devptr = undistorted_depth.dev_ptr;

  match_parameter.setDevData();

  // depth_fuse_initialize(match_parameter);
}

rmd::SeedMatrix::~SeedMatrix()
{
  for(int i = 0; i < keyframe_list_length; i++)
  {
    delete keyframe_list[i].frame_hostptr;
    keyframe_list[i].frame_hostptr = NULL;
  }
  if(has_fisheye_table)
  {
    cudaFreeArray(angleToR_array);
    cudaFreeArray(rToAngle_array);
  }
}

void rmd::SeedMatrix::set_fisheye(
      const std::vector<double> &a2r,
      const std::vector<double> &r2a,
      const FisheyeParam fisheyepara)
{
  const int table_size = a2r.size();
  cudaMallocArray (&angleToR_array, &angle2r_tex.channelDesc, table_size, 1);
  cudaMallocArray (&rToAngle_array, &r2angle_tex.channelDesc, table_size, 1);

  float* table;
  table = (float*)malloc(table_size * sizeof(float));
  for(int i = 0; i < table_size; i++)
  {
    table[i] = a2r[i];
  }
  cudaMemcpyToArray(angleToR_array, 0, 0, table, table_size * sizeof(float), cudaMemcpyHostToDevice);
  for(int i = 0; i < table_size; i++)
  {
    table[i] = r2a[i];
  }
  cudaMemcpyToArray(rToAngle_array, 0, 0, table, table_size * sizeof(float), cudaMemcpyHostToDevice);
  bindTextureArray(r2angle_tex, rToAngle_array, cudaFilterModeLinear);
  bindTextureArray(angle2r_tex, angleToR_array, cudaFilterModeLinear);
  has_fisheye_table = true;

  match_parameter.fisheye_para = fisheyepara;
  match_parameter.setDevData();

  //mask
  mask_generate<<<dim_grid_image, dim_block_image>>>(fisheyepara, fisheye_mask.dev_ptr, pixel_dir.dev_ptr);
  cudaDeviceSynchronize();
  bindTexture(fisheye_mask_tex, fisheye_mask, cudaFilterModePoint);

  //set the undistort map
  dim3 map_block;
  dim3 map_grid;
  map_block.x = 32;
  map_block.y = 32;
  map_grid.x = (undistorted_map.width + map_block.x - 1) / map_block.x;
  map_grid.y = (undistorted_map.height + map_block.y - 1) / map_block.y;
  initial_undistort<<<map_grid, map_block>>>(fisheyepara, undistorted_map.dev_ptr, UNDISTORT_STEP);
  cudaDeviceSynchronize();
}

bool rmd::SeedMatrix::is_keyframe(const rmd::SE3<float> &T_curr_world)
{
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
    if(trans_length > 0.05f)
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
      cv::Mat _add_mat,
      float *host_img_align_row_maj,
      const SE3<float> T_curr_world)
{
  std::clock_t start = std::clock();
  frame_index++;
  const SE3<float> current_pose = T_curr_world.inv();
  _add_mat.copyTo(ref_mat);
  curr_img.setDevData(host_img_align_row_maj);
  printf("\n\n\nnew frame %d, has keyframe %d!\n", frame_index, keyframe_list.size());

  bool add_keyframe = is_keyframe(T_curr_world);
  if(add_keyframe)
  {
    frame_element new_element;
    new_element.frame_hostptr = new DeviceImage<float>(width, height);
    new_element.world_pose = T_curr_world;
    new_element.frame_hostptr->setDevData(host_img_align_row_maj);
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
  match_parameter.current_worldpose = T_curr_world;
  match_parameter.frame_nums++;

  rmd::bindTexture(curr_img_tex, curr_img, cudaFilterModeLinear);

  printf("till prepare the image cost %f ms.\n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);

  match_parameter.current_to_key = first_frame_pose * T_curr_world.inv();
  match_parameter.setDevData();
  printf("till prepare the keyframe cost %f ms \n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);

  //match the frame to the key frame
  match_features(match_parameter, add_keyframe);
  printf("till feature match cost %f ms \n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);

  undistorted_depthmap();
  undistorted_reference();

  depth_fuse(match_parameter);

  cudaUnbindTexture(curr_img_tex);

  match_parameter.previous2_worldpose = match_parameter.previous_worldpose;
  match_parameter.previous_worldpose = match_parameter.current_worldpose;
  return true;
}

void rmd::SeedMatrix::get_depthmap(float *host_denoised)
{
  std::clock_t start = std::clock();
  depth_output.getDevData(host_denoised);
  printf("download depth map cost %f ms \n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);
}

void rmd::SeedMatrix::get_pointcloud(float *host_pointcloud)
{
  std::clock_t start = std::clock();
  SE3<float> cur_to_world = match_parameter.current_worldpose.inv();
  generate_cloud_with_intensity(cur_to_world, undistorted_ref, undistorted_depth, pub_cloud);
  pub_cloud.getDevData( reinterpret_cast<float4*> (host_pointcloud) );
  printf("download pointcloud cost %f ms \n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);
}


void rmd::SeedMatrix::undistorted_depthmap()
{
  std::clock_t start = std::clock();
  rmd::bindTexture(depth_img_tex, depth_output, cudaFilterModePoint);
  dim3 map_block;
  dim3 map_grid;
  map_block.x = 32;
  map_block.y = 32;
  map_grid.x = (undistorted_map.width + map_block.x - 1) / map_block.x;
  map_grid.y = (undistorted_map.height + map_block.y - 1) / map_block.y;
  undistort_depth<<<map_grid, map_block>>>(undistorted_map.dev_ptr, undistorted_depth.dev_ptr);
  cudaUnbindTexture(depth_img_tex);
  printf("undistorte depth cost %f ms \n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);
}

void rmd::SeedMatrix::get_undistorted_depthmap(float *host_reference)
{
  std::clock_t start = std::clock();
  undistorted_depth.getDevData(host_reference);
  printf("download undistorted depth cost %f ms \n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);
}

void rmd::SeedMatrix::undistorted_reference()
{
  std::clock_t start = std::clock();
  dim3 map_block;
  dim3 map_grid;
  map_block.x = 32;
  map_block.y = 32;
  map_grid.x = (undistorted_map.width + map_block.x - 1) / map_block.x;
  map_grid.y = (undistorted_map.height + map_block.y - 1) / map_block.y;
  undistort_ref<<<map_grid, map_block>>>(undistorted_map.dev_ptr, undistorted_ref.dev_ptr);
  printf("undistorte reference cost %f ms \n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);
}
void rmd::SeedMatrix::get_ReferenceImage(float *host_undistorted)
{
  std::clock_t start = std::clock();
  undistorted_ref.getDevData(host_undistorted);
  printf("download undistorted reference cost %f ms \n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);
}

const rmd::DeviceImage<float> & rmd::SeedMatrix::getRefImg()const
{
  return ref_img;
}
