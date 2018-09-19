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

#ifndef RMD_TEXTURE_MEMORY_CUH
#define RMD_TEXTURE_MEMORY_CUH

#include <cuda_runtime.h>
#include <rmd/device_image.cuh>
#include <rmd/device_linear.cuh>
// #include <iostream>

namespace rmd
{

texture<float, cudaTextureType2D, cudaReadModeElementType> ref_img_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> depth_img_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> curr_img_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> curr_img_l1_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> curr_img_l2_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> last_img_tex;

texture<float, cudaTextureType2D, cudaReadModeElementType> ref_x_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> ref_y_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> curr_x_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> curr_y_tex;

//for key frame
texture<float, cudaTextureType2D, cudaReadModeElementType> keyframe_tex;

texture<float, cudaTextureType2D, cudaReadModeElementType> g_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> sigma_tex;

//fisheye look up table
texture<float, cudaTextureType1D, cudaReadModeElementType> r2angle_tex;
texture<float, cudaTextureType1D, cudaReadModeElementType> angle2r_tex;
texture<uchar, cudaTextureType2D, cudaReadModeElementType> fisheye_mask_tex;

template<typename ElementType>
inline void bindTexture(
    texture<ElementType, cudaTextureType2D> &tex,
    const DeviceImage<ElementType> &mem,
    cudaTextureFilterMode filter_mode=cudaFilterModeLinear)
{
  tex.addressMode[0] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.addressMode[1] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.filterMode = filter_mode;
  tex.normalized = false;

  const cudaError bindStatus = cudaBindTexture2D(
        0,
        tex,
        mem.data,
        mem.getCudaChannelFormatDesc(),
        mem.width,
        mem.height,
        mem.pitch
        );
  if(bindStatus != cudaSuccess)
  {
    throw CudaException("Unable to bind texture: ", bindStatus);
  }
}

template<typename ElementType>
inline void bindTexture(
    texture<ElementType, cudaTextureType1D> &tex,
    const DeviceLinear<ElementType> &mem,
    cudaTextureFilterMode filter_mode=cudaFilterModePoint)
{
  tex.addressMode[0] = cudaAddressModeClamp; // Neumann Boundary Conditions
  // tex.addressMode[1] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.filterMode = filter_mode;
  tex.normalized = false;

  const cudaError bindStatus = cudaBindTexture(
        0,
        tex,
        mem.data,
        mem.getCudaChannelFormatDesc(),
        mem.length * sizeof(ElementType)
        );
  if(bindStatus != cudaSuccess)
  {
    throw CudaException("Unable to bind linear texture: ", bindStatus);
  }
}

template<typename ElementType>
inline void bindTextureArray(
    texture<ElementType, cudaTextureType1D> &tex,
    const cudaArray* cuarray,
    cudaTextureFilterMode filter_mode=cudaFilterModeLinear)
{
  tex.addressMode[0] = cudaAddressModeClamp; // Neumann Boundary Conditions
  // tex.addressMode[1] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.filterMode = filter_mode;
  tex.normalized = false;

  const cudaError bindStatus = cudaBindTextureToArray(
        tex,cuarray);
  if(bindStatus != cudaSuccess)
  {
    throw CudaException("Unable to bind array texture: ", bindStatus);
  }
}

} // rmd namespace

#endif // RMD_TEXTURE_MEMORY_CUH
