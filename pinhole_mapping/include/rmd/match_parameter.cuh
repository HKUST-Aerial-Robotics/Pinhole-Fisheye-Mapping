#ifndef MATCH_PARAMETER_CUH
#define MATCH_PARAMETER_CUH

#include <vector_types.h>
#include <cuda_toolkit/helper_math.h>
#include <rmd/se3.cuh>
#include <rmd/camera_model/pinhole_camera.cuh>
 
#include <rmd/pixel_cost.cuh>
#include <rmd/device_linear.cuh>
// #include <rmd/tracked_feature.cuh>
namespace rmd
{

struct MatchParameter
{
  __host__
  MatchParameter( int _image_width,
                  int _image_heigth):
  keypoint_step(4),
  image_width(_image_width),
  image_height(_image_heigth),
  is_dev_allocated(false),
  min_keypoint_match_count(5),
  consist_check_distence(3),
  consisit_support_window(10),
  consisit_support_number(10),
  consisit_support_threshold(10),
  sample_grid_size(16),
  disparity_sample_num(DEPTH_NUM),
  max_match_dist(50),
  min_match_dist(0.5),
  inv_depth_step((1.0f / 0.5f - 1.0f / 50.0f)/ (float)(DEPTH_NUM - 1)),
  sigma(1),
  sigma_radius(1),
  gamma(5),
  beta(10.2),
  pixel_match_halfsize(2)
  {
    grids_width = (int) ceil((float)image_width / (float)sample_grid_size);
    grids_height = (int) ceil((float)image_height / (float)sample_grid_size);
  }

  void set_image_devptr(  DeviceImage<float> *_ref_image_devptr,
                          DeviceImage<float> *_curr_image_devptr,
                          DeviceImage<float> *_depth_map_devptr)
  {
    ref_image_devptr = _ref_image_devptr;
    current_image_devptr = _curr_image_devptr;
    depth_map_devptr = _depth_map_devptr;
  }

  void set_transform(rmd::SE3<float> _transform)
  {
    transform = _transform;
    inverse_transform = _transform.inv();
  }

  void set_camera_devptr(PinholeCamera *camera)
  {
    camera_devptr = camera;
  }

  __host__
  void setDevData()
  {
    if(!is_dev_allocated)
    {
      // Allocate device memory
      const cudaError err = cudaMalloc(&dev_ptr, sizeof(*this));
      if(err != cudaSuccess)
        throw CudaException("PinholeCamera, cannot allocate device memory to store image parameters.", err);
      else
      {
        is_dev_allocated = true;
      }
    }
    // Copy data to device memory
    const cudaError err2 = cudaMemcpy(dev_ptr, this, sizeof(*this), cudaMemcpyHostToDevice);
    if(err2 != cudaSuccess)
      throw CudaException("PinholeCamera, cannot copy image parameters to device memory.", err2);
  }

  int image_width;
  int image_height;
  int keypoint_step;
  float max_invdepth;
  int min_keypoint_match_count;
  rmd::SE3<float> transform;
  rmd::SE3<float> inverse_transform;
  PinholeCamera *camera_devptr;

  //
  float min_match_dist;
  float max_match_dist;

  //image list information
  int frame_nums;
  rmd::SE3<float> current_worldpose; //current to world

  //match 
  SE3<float> current_to_key;

  // for keypint left-right check
  int consist_check_distence;

  //for keypoint support check
  int consisit_support_window;
  int consisit_support_number;
  int consisit_support_threshold;

  //sample grid size
  int grids_width;
  int grids_height;
  int sample_grid_size;
  int disparity_sample_num;
  float inv_depth_step;

  //pixel match parameter
  float sigma;
  int sigma_radius;
  float gamma;
  float beta;
  int pixel_match_halfsize;

  DeviceImage<float> *previous2_depth_map_devptr;
  DeviceImage<float> *previous_depth_map_devptr;
  DeviceImage<float> *depth_map_devptr;
  DeviceImage<float> *previous2_depth_map_hostptr;
  DeviceImage<float> *previous_depth_map_hostptr;
  DeviceImage<float> *depth_map_hostptr;
  SE3<float> previous_worldpose;
  SE3<float> previous2_worldpose;

  DeviceImage<int> *prefeature_age_hostptr;
  DeviceImage<int> *prefeature_age_devptr;
  DeviceImage<int> *curfeature_age_hostptr;
  DeviceImage<int> *curfeature_age_devptr;

  DeviceImage<float4> *pre_fuse_hostptr;
  DeviceImage<float4> *pre_fuse_devptr;
  DeviceImage<float4> *cur_fuse_hostptr;
  DeviceImage<float4> *cur_fuse_devptr;

  DeviceImage<float> *depth_output_devptr;
  
  DeviceImage<float> *ref_image_devptr;
  DeviceImage<float> *current_image_devptr;
  DeviceImage<float> *current_image_hostptr;

  DeviceImage<float> *keyframe_devptr[60];
  float keyframe_theta[60]; //describe the rotation of each keyframe
  SE3<float> keyframe_worldpose[60];

  MatchParameter *dev_ptr;
  bool is_dev_allocated;
};

} // namespace rmd

#endif