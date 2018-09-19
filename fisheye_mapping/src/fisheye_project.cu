#include <cuda_toolkit/helper_math.h>
#include <rmd/texture_memory.cuh>
#include <rmd/device_image.cuh>
#include <rmd/camera_model/fisheye_param.cuh>
#include <ctime>

namespace rmd
{
#define THETA_MAX 90.0f / 180.0f * M_PI
#define MARGIN 3

//function declear here!
inline  __device__
float3 cam2world(const float2 & uv, const FisheyeParam &param);
inline  __device__
float2 world2cam(const float3 & xyz, const FisheyeParam &param);
inline  __device__
float get_theta(const float3 &uv, const FisheyeParam &param);
inline __device__
bool is_vaild(const int2 &uv, const FisheyeParam &param);
inline __device__
bool is_vaild(const int &x, const int &y, const FisheyeParam &param);
inline __device__
bool is_vaild(const int &x, const int &y);
__global__
void mask_generate(FisheyeParam param, DeviceImage<uchar> *mask_ptr, DeviceImage<float3> *pixel_dir);

//function define
inline  __device__
float3 cam2world(const float2 & uv, const FisheyeParam &param)
{
	float2 p = make_float2(	uv.x * param.inv_k[0] + uv.y * param.inv_k[1] + param.inv_k[2],
													uv.y * param.inv_k[3] + param.inv_k[4]);
	float theta, phi;

	float r = length(p);
	if(r < 1e-10)
		phi = 0.0f;
	else
		phi = atan2f(p.y, p.x);

	float num = r / param.diff_r;
	theta = tex1D( r2angle_tex, num + 0.5f );

	return make_float3(	cosf(phi) * sinf(theta),
											sinf(phi) * sinf(theta),
											cosf(theta));
}

inline  __device__
float2 world2cam(const float3 & xyz, const FisheyeParam &param)
{
	float theta = acosf(xyz.z / length(xyz));
	float phi = atan2f(xyz.y, xyz.x);
	float r;
	if(theta < param.maxIncidentAngle && theta > 1e-10)
	{
		float num = theta / param.diff_angle;
		r = tex1D(angle2r_tex, num + 0.5);
	}
	else
		r = 0.0f;
	float2 p_u;
	p_u.x = r * cosf(phi);
	p_u.y = r * sinf(phi);
	return make_float2(	param.A11 * p_u.x + param.A12 * p_u.y + param.u0,
											param.A22 * p_u.y + param.v0);
}

inline  __device__
float get_theta(const float2 &uv, const FisheyeParam &param)
{
	float2 p = make_float2(	uv.x * param.inv_k[0] + uv.y * param.inv_k[1] + param.inv_k[2],
													uv.y * param.inv_k[3] + param.inv_k[4]);

	float r = length(p);
	float num = r / param.diff_r;
	return tex1D( r2angle_tex, num + 0.5f );
}

__global__
void mask_generate(FisheyeParam param, DeviceImage<uchar> *mask_ptr, DeviceImage<float3> *pixel_dir)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x >= param.width || y >= param.height)
		return;

	float theta = get_theta(make_float2((float)x, (float)y), param);

	if (theta > THETA_MAX || x < MARGIN || y < MARGIN || x > (param.width - MARGIN) || y > (param.height - MARGIN))
	{
		mask_ptr->atXY(x,y) = 0;
		pixel_dir->atXY(x,y) = make_float3(0.0f, 0.0f, 0.0f);
	}
	else
	{
		float3 feature_dir = cam2world(make_float2((float)x, (float)y), param);
		pixel_dir->atXY(x,y) = feature_dir;
		mask_ptr->atXY(x,y) = 1;
	}
}

__global__ void initial_undistort(FisheyeParam param, DeviceImage<int2> *undistort_table_devptr, int step)
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	const int x_center = undistort_table_devptr->width / 2;
	const int y_center = undistort_table_devptr->height / 2;

	const int x_cam = x - x_center;
	const int y_cam = y - y_center;
	float theta_deg = sqrtf(x_cam * x_cam + y_cam * y_cam) / (float) step;
	if(theta_deg > 90.0f)
	{
		undistort_table_devptr->atXY(x,y) = make_int2(-1, -1);
		return;
	}
	else
	{
		float phi_rad = atan2f((float)y_cam, (float)x_cam);
		float theta_rad = theta_deg * 0.0174533;
		float3 ray = make_float3(sinf(theta_rad) * cosf(phi_rad), sinf(theta_rad) * sinf(phi_rad), cosf(theta_rad));
		int2 undistorted = make_int2( world2cam(ray, param) );
		if ( is_vaild(undistorted, param) )
		{
			undistort_table_devptr->atXY(x,y) = undistorted;
			return;
		}
		else
			undistort_table_devptr->atXY(x,y) = make_int2(-1, -1);
	}
}

__global__ void undistort_depth(DeviceImage<int2> *map_devptr, DeviceImage<float> *undist_devptr)
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if(x >= map_devptr->width || y >= map_devptr->height)
		return;

	int2 map = map_devptr->atXY(x,y);

	if(map.x < 0 || map.y < 0)
	{
		undist_devptr->atXY(x,y) = -1.0f;
		return;
	}

	undist_devptr->atXY(x,y) = tex2D(depth_img_tex, map.x, map.y);
}

__global__ void undistort_ref(DeviceImage<int2> *map_devptr, DeviceImage<float> *undist_devptr)
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if(x >= map_devptr->width || y >= map_devptr->height)
		return;

	int2 map = map_devptr->atXY(x,y);

	if(map.x < 0 || map.y < 0)
	{
		undist_devptr->atXY(x,y) = -1.0f;
		return;
	}

	undist_devptr->atXY(x,y) = tex2D(curr_img_tex, map.x, map.y);
}
//debug
__global__
void debug_dev(FisheyeParam param, DeviceImage<float> *dev_ptr)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	float theta = get_theta(make_float2((float)x, (float)y), param);

	if (theta > THETA_MAX || x < MARGIN || y < MARGIN || x > (param.width - MARGIN) || y > (param.height - MARGIN))
	{
		dev_ptr->atXY(x,y) = 0.0f;
	}
	else
	{
		theta *= 100;
		float up_theta = get_theta(make_float2((float)x, (float)y - 5), param) * 100;
		float down_theta = get_theta(make_float2((float)x, (float)y + 5), param) * 100;
		float left_theta = get_theta(make_float2((float)x - 5, (float)y), param) * 100;
		float right_theta = get_theta(make_float2((float)x + 5, (float)y), param) *100;
		float diff = (up_theta - theta) * (up_theta - theta) + (down_theta - theta) * (down_theta - theta)
									+ (left_theta - theta) * (left_theta - theta) + (right_theta - theta) * (right_theta - theta);
		diff = sqrtf(diff);
		dev_ptr->atXY(x,y) = diff;
	}
}


inline __device__
bool is_vaild(const int2 &uv, const FisheyeParam &param)
{
	if(uv.x < MARGIN || uv.y < MARGIN || uv.x >= (param.width - MARGIN) || uv.y >= (param.height - MARGIN))
		return false;
	return (bool)tex2D(fisheye_mask_tex, (int)uv.x, (int)uv.y);
}

inline __device__
bool is_vaild(const int &x, const int &y, const FisheyeParam &param)
{
	if(x < MARGIN || y < MARGIN || x >= (param.width - MARGIN) || y >= (param.height - MARGIN))
		return false;
	return (bool)tex2D(fisheye_mask_tex, (int)x, (int)y);
}

inline __device__
bool is_vaild(const int &x, const int &y)
{
	return (bool)tex2D(fisheye_mask_tex, (int)x, (int)y);
}

//debug
__global__
void mask_paint(FisheyeParam param, DeviceImage<float> *image_ptr, DeviceImage<float> *ref_ptr)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(x >= param.width || y >= param.height)
		return;

	uchar vaild = tex2D(fisheye_mask_tex, (float)x, (float)y);

	if(vaild)
		image_ptr->atXY(x,y) = ref_ptr->atXY(x,y);
	else
		image_ptr->atXY(x,y) = 0;
}

__global__
void read_tex()
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	float value[5];
	for(int i = 0; i < 5; i++)
	{
		float x_i = (float) x + i * 0.2f;
		// value[i] = tex1Dfetch(r2angle_tex, x_i);
		value[i] = tex1D(r2angle_tex, x_i + 0.5f);
	}
	printf("from %d, get %f, %f, %f, %f, %f\n",
		x,
		value[0],
		value[1],
		value[2],
		value[3],
		value[4]);
}

__global__
void project_test(const FisheyeParam param)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	float2 image_point = make_float2(x * 10, y * 10);
	float3 space_point = cam2world(image_point, param);
	float2 back_point = world2cam(space_point, param);

	printf(" the point (%f, %f) -> (%f, %f, %f) -> (%f, %f).\n",
		image_point.x, image_point.y,
		space_point.x, space_point.y, space_point.z,
		back_point.x, back_point.y);
}

}