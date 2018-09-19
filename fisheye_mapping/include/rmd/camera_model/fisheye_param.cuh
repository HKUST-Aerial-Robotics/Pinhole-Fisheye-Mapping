#ifndef RMD_FISHEYE_PARAM_CUH_
#define RMD_FISHEYE_PARAM_CUH_

#include <vector_types.h>
#include <cuda_toolkit/helper_math.h>

namespace rmd
{
  
struct FisheyeParam
{
  float inv_k[5];
  float diff_r;
  float diff_angle;
  float maxIncidentAngle;
  float A11, A12, A22;
  float u0, v0;
  int width, height;
};

} // namespace rmd


#endif // RMD_FISHEYE_PARAM_CUH_
