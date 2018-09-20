#ifndef DEVICE_LINEAR_CUH
#define DEVICE_LINEAR_CUH

#include <assert.h>
#include <cuda_runtime.h>
#include <rmd/cuda_exception.cuh>

namespace rmd
{
template<typename ElementType>
struct DeviceLinear
{
  __host__
  DeviceLinear(size_t _length)
    : length(_length)
  {
    cudaError err = cudaMalloc(&data, _length * sizeof(ElementType));
    if(err != cudaSuccess)
      throw CudaException("DeviceLinear: unable to allocate linear memory.", err);

    err = cudaMalloc(
          &dev_ptr,
          sizeof(*this));
    if(err != cudaSuccess)
      throw CudaException("DeviceLinear: cannot allocate device memory to store linear parameters.", err);

    err = cudaMemcpy(
          dev_ptr,
          this,
          sizeof(*this),
          cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
      throw CudaException("DeviceLinear: cannot copy linear parameters to device memory.", err);
  }

  __device__
  ElementType & operator()(size_t x)
  {
    return at(x);
  }

  __device__
  const ElementType & operator()(size_t x) const
  {
    return at(x);
  }

  __device__
  ElementType &at(size_t x)
  {
    return data[x];
  }

  __device__
  const ElementType &at(size_t x) const
  {
    return data[x];
  }

  /// Upload aligned_data_row_major to device memory
  __host__
  void setDevData(const ElementType *aligned_data)
  {
    const cudaError err = cudaMemcpy(
          data,
          aligned_data,
          length * sizeof(ElementType),
          cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
      throw CudaException("DeviceLinear: unable to copy data from host to device.", err);
  }

  /// Download the data from the device memory to aligned_data_row_major, a preallocated array in host memory
  __host__
  void getDevData(ElementType* aligned_data) const
  {
    const cudaError err = cudaMemcpy(
          data,
          aligned_data,
          length * sizeof(ElementType),
          cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)
    {
      throw CudaException("DeviceLinear: unable to copy data from device to host.", err);
    }
  }

  __host__
  ~DeviceLinear()
  {
    cudaError err = cudaFree(data);
    if(err != cudaSuccess)
      throw CudaException("DeviceLinear: unable to free allocated memory.", err);
    err = cudaFree(dev_ptr);
    if(err != cudaSuccess)
      throw CudaException("DeviceLinear: unable to free allocated memory.", err);
  }

  __host__
  cudaChannelFormatDesc getCudaChannelFormatDesc() const
  {
    return cudaCreateChannelDesc<ElementType>();
  }

  __host__
  void zero()
  {
    const cudaError err = cudaMemset(
          data,
          0,
          length*sizeof(ElementType));
    if(err != cudaSuccess)
      throw CudaException("DeviceLinear: unable to zero.", err);
  }

  __host__
  DeviceLinear<ElementType> & operator= (const DeviceLinear<ElementType> &other_linear)
  {
    if(this != & other_linear)
    {
      assert(length  == other_linear.length);
      const cudaError err = cudaMemcpy( data,
                                        other_linear.data,
                                        length * sizeof(ElementType),
                                        cudaMemcpyDeviceToDevice);
      if(err != cudaSuccess)
        throw CudaException("DeviceLinear, operator '=': unable to copy data from another linear.", err);
    }
    return *this;
  }

  // fields
  size_t length;
  ElementType *data;
  DeviceLinear<ElementType> *dev_ptr;
};

} // namespace rmd

#endif // DEVICE_LINEAR_CUH
