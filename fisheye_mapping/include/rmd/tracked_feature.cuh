#ifndef TRACKED_FEATURE_CUH
#define TRACKED_FEATURE_CUH

#include <vector_types.h>
#include <cuda_toolkit/helper_math.h>
#include <stdio.h>

struct TrackedFeature
{
	float x,y,z;
	float baseline;
	TrackedFeature():baseline(-1){}
};


#endif // TRACKED_FEATURE_CUH