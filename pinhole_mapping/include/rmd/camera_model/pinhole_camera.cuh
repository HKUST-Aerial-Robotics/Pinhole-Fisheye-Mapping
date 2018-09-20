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

#ifndef RMD_PINHOLE_CAMERA_CUH_
#define RMD_PINHOLE_CAMERA_CUH_

#include <vector_types.h>
#include <cuda_toolkit/helper_math.h>

namespace rmd
{
  
struct PinholeCamera
{
  __host__
  PinholeCamera()
    : fx(0.0f), fy(0.0f), cx(0.0f), cy(0.0f)
  {
    is_dev_allocated = false;
  }

  __host__ __device__
  PinholeCamera(const PinholeCamera& camera_ref)
  :fx(camera_ref.fx), fy(camera_ref.fy), cx(camera_ref.cx), cy(camera_ref.cy)
  {
    is_dev_allocated = false;
  }

  __host__ __device__
  PinholeCamera(float fx, float fy,
                float cx, float cy)
    : fx(fx), fy(fy), cx(cx), cy(cy)
  {}

  __host__ __device__
  PinholeCamera& operator= (const PinholeCamera &other_camera)
  {
    if(this != &other_camera)
    {
      fx = other_camera.fx;
      fy = other_camera.fy;
      cx = other_camera.cx;
      cy = other_camera.cy;
    }
    return *this;
  }

  __host__ __device__ __forceinline__
  float3 cam2world(const float2 & uv) const
  {
    return make_float3((uv.x - cx)/fx,
                       (uv.y - cy)/fy,
                       1.0f);
  }

  __host__ __device__ __forceinline__
  float2 world2cam(const float3 & xyz) const
  {
    return make_float2(fx*xyz.x / xyz.z + cx,
                       fy*xyz.y / xyz.z + cy);
  }

  __host__ __device__ __forceinline__
  float3 world2cam_f3(const float3 & xyz) const
  {
    return make_float3(fx*xyz.x / xyz.z + cx,
                       fy*xyz.y / xyz.z + cy,
                       xyz.z);
  }

  __host__ __device__ __forceinline__
  float getOnePixAngle() const
  {
    return atan2f(1.0f, 2.0f*fx)*2.0f;
  }

    __host__ __device__ __forceinline__
  float getf() const
  {
    return (fx + fy) / 2;
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

  float fx, fy;
  float cx, cy;
  bool is_dev_allocated;
  PinholeCamera *dev_ptr;
};

} // namespace rmd


#endif // RMD_PINHOLE_CAMERA_CUH_
