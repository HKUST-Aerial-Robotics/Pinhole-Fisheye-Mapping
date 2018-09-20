#include <cuda_toolkit/helper_math.h>
#include <rmd/device_image.cuh>
#include <rmd/texture_memory.cuh>
#include <rmd/match_parameter.cuh>
#include <rmd/pixel_cost.cuh>
#include <ctime>

namespace rmd
{
//function declear here!
void hbp(DeviceImage<PIXEL_COST> &cost_map, DeviceImage<float> &depth);
__global__ void cost_distribute( 	DeviceImage<PIXEL_COST> *l0_cost_devptr,
																	DeviceImage<PIXEL_COST> *l1_cost_devptr);
__global__ void bp1(	DeviceImage<PIXEL_COST> *data_devptr,
											DeviceImage<PIXEL_COST> *lm_devptr,
											DeviceImage<PIXEL_COST> *rm_devptr,
											DeviceImage<PIXEL_COST> *up_devptr,
											DeviceImage<PIXEL_COST> *dm_devptr,
											bool A_set);
__global__ void upsample(	DeviceImage<PIXEL_COST> *l1_message_devptr,
													DeviceImage<PIXEL_COST> *l0_message_devptr);
__global__ void depth_filter(	DeviceImage<PIXEL_COST> *data_devptr,
															DeviceImage<PIXEL_COST> *lm_devptr,
															DeviceImage<PIXEL_COST> *rm_devptr,
															DeviceImage<PIXEL_COST> *up_devptr,
															DeviceImage<PIXEL_COST> *dm_devptr,
															DeviceImage<float> *depth_devptr);

//function define here!
void hbp(DeviceImage<PIXEL_COST> &cost_map, DeviceImage<float> &depth)
{
	const int width = cost_map.width;
	const int height = cost_map.height;
	const int hbp_level = 4;
	// int hbp_iterate[5] = {5, 5, 10, 10, 0}; // from fine to coarse 4 level
	int hbp_iterate[5] = {5, 10, 10, 10, 0}; // from fine to coarse 3 level

	int h_width[5]; //next four level
	int h_height[5]; //next four level

	h_width[0] = width;
	h_height[0] = height;

	for(int i = 1; i < hbp_level; i++)
	{
		h_width[i] = (h_width[i - 1] + 1) / 2;
		h_height[i] = (h_height[i - 1] + 1) / 2;
	}

	//create the hierarchical cost map
  std::clock_t start = std::clock();
	DeviceImage<PIXEL_COST> *prycost_hostptr[5];
	prycost_hostptr[0] = &cost_map;
	for(int i = 1; i < hbp_level; i++)
	{
		prycost_hostptr[i] = new DeviceImage<PIXEL_COST>(h_width[i], h_height[i]);
	}
	dim3 hier_block;
	dim3 hier_grid;
	hier_block.x = 1;
	hier_block.y = 1;
	hier_block.z = DEPTH_NUM;
	for(int i = 1; i < hbp_level; i++)
	{
		hier_grid.x = (h_width[i] + hier_block.x - 1) / hier_block.x;
		hier_grid.y = (h_height[i] + hier_block.y - 1) / hier_block.y;
		cost_distribute<<<hier_grid, hier_block>>>(
			prycost_hostptr[i - 1]->dev_ptr,
			prycost_hostptr[i]->dev_ptr);
  	cudaDeviceSynchronize();
	}
  printf("   cost distribute cost %f ms\n", ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);

	//loopy bp on each level
	//create the message four dirs
	DeviceImage<PIXEL_COST> *message_hostptr[4];
	message_hostptr[0] = new DeviceImage<PIXEL_COST>(h_width[hbp_level - 1], h_height[hbp_level - 1]);
	message_hostptr[1] = new DeviceImage<PIXEL_COST>(h_width[hbp_level - 1], h_height[hbp_level - 1]);
	message_hostptr[2] = new DeviceImage<PIXEL_COST>(h_width[hbp_level - 1], h_height[hbp_level - 1]);
	message_hostptr[3] = new DeviceImage<PIXEL_COST>(h_width[hbp_level - 1], h_height[hbp_level - 1]);
	message_hostptr[0]->zero();
	message_hostptr[1]->zero();
	message_hostptr[2]->zero();
	message_hostptr[3]->zero();

	for(int i_leverl = hbp_level - 1; i_leverl >= 0; i_leverl--)
	{
		start = std::clock();
		//if i_leverl is not the coarsest, initialize the message
		if( i_leverl < (hbp_level - 1) )
		{
			dim3 message_up_block;
			dim3 message_up_grid;
			message_up_block.x = DEPTH_NUM;
			message_up_grid.x = h_width[i_leverl + 1];
			message_up_grid.y = h_height[i_leverl + 1];

			DeviceImage<PIXEL_COST> *message_next_hostptr[4];
			message_next_hostptr[0] = new DeviceImage<PIXEL_COST>(h_width[i_leverl], h_height[i_leverl]);
			message_next_hostptr[1] = new DeviceImage<PIXEL_COST>(h_width[i_leverl], h_height[i_leverl]);
			message_next_hostptr[2] = new DeviceImage<PIXEL_COST>(h_width[i_leverl], h_height[i_leverl]);
			message_next_hostptr[3] = new DeviceImage<PIXEL_COST>(h_width[i_leverl], h_height[i_leverl]);

			for(int mess_i = 0; mess_i < 4; mess_i++)
				upsample<<<message_up_grid, message_up_block>>>(
							message_hostptr[mess_i]->dev_ptr,
							message_next_hostptr[mess_i]->dev_ptr);

			cudaDeviceSynchronize();

			for(int mess_i = 0; mess_i < 4; mess_i++)
			{
				delete message_hostptr[mess_i];
				message_hostptr[mess_i] = message_next_hostptr[mess_i];
			}
		}

		//loopy bp
		dim3 bp_block;
		dim3 bp_grid;
		bp_block.x = 4;
		bp_block.y = DEPTH_NUM;
		bp_grid.x = h_width[i_leverl];
		bp_grid.y = h_height[i_leverl];
		bp_grid.x = (bp_grid.x + 1) / 2; //every iterate on the A or B set of the whole image
		bool A_set = true;
		for(int i_iterate = 0; i_iterate < hbp_iterate[i_leverl]; i_iterate++)
		{
			bp1<<<bp_grid, bp_block>>>(
					prycost_hostptr[i_leverl]->dev_ptr,
					message_hostptr[0]->dev_ptr,
					message_hostptr[1]->dev_ptr,
					message_hostptr[2]->dev_ptr,
					message_hostptr[3]->dev_ptr,
					A_set);
			A_set = !A_set;
			cudaDeviceSynchronize();
		}
  	printf("   loop %d (%d, %d) %d times cost %f ms\n",
  			i_leverl,
  			h_width[i_leverl], h_height[i_leverl],
  			hbp_iterate[i_leverl],
  			( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000);
	}

	//selete the min cost depth
	dim3 depth_block;
	dim3 depth_grid;
	depth_block.x = DEPTH_NUM / 2;
	depth_grid.x = h_width[0];
	depth_grid.y = h_height[0];
	depth_filter<<<depth_grid, depth_block>>>(
		prycost_hostptr[0]->dev_ptr,
		message_hostptr[0]->dev_ptr,
		message_hostptr[1]->dev_ptr,
		message_hostptr[2]->dev_ptr,
		message_hostptr[3]->dev_ptr,
		depth.dev_ptr);

	for(int i = 1; i < hbp_level; i++)
	{
		delete prycost_hostptr[i];
	}
	delete message_hostptr[0];
	delete message_hostptr[1];
	delete message_hostptr[2];
	delete message_hostptr[3];
}

__global__ void cost_distribute(DeviceImage<PIXEL_COST> *l0_cost_devptr,
																DeviceImage<PIXEL_COST> *l1_cost_devptr)
{
	const int width = l1_cost_devptr->width;
	const int height = l1_cost_devptr->height;
	const int l0_width = l0_cost_devptr->width;
	const int l0_height = l0_cost_devptr->height;

  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int depth_id = threadIdx.z;

  if(x >= width || y >= height)
  	return;

  float cost_sum(0.0f);
  // int cost_num(0);
  for(int i = 0; i < 2; i++)
  {
  	for(int j = 0; j < 2; j++)
  	{
  		if( (2 * x + i) < l0_width && (2 * y + j) < l0_height)
  		{
  			cost_sum += (l0_cost_devptr->atXY((2 * x + i), (2 * y + j))).get_cost(depth_id);
  			// cost_num ++;
  		}
  	}
  }

  // cost_sum = cost_sum / (float) cost_num;
  (l1_cost_devptr->atXY(x,y)).set_cost(depth_id,cost_sum);
}

__global__ void bp1(	DeviceImage<PIXEL_COST> *data_devptr,
											DeviceImage<PIXEL_COST> *lm_devptr,
											DeviceImage<PIXEL_COST> *rm_devptr,
											DeviceImage<PIXEL_COST> *up_devptr,
											DeviceImage<PIXEL_COST> *dm_devptr,
											bool A_set)
{
	const int width = data_devptr->width;
	const int height = data_devptr->height;
	int x = blockIdx.x;
	int y = blockIdx.y;
  int dir = threadIdx.x;
  int depth_id = threadIdx.y;

  //this works well in fly area
  float P1;
  float P2;

	//this is used in gernel
//   P1 = 0.5f;
//   P2 = 1.5f;

	//this is used in gernel
//   P1 = 0.3f;
//   P2 = 2.0f;//6.45
	P1 = 0.2f;
	P2 = 2.0f;//6.19
	
  if(A_set)
  	x = x * 2 + y % 2;
  else
  	x = x * 2 + (y + 1) % 2 ;

  if(x >= width || y >= height)
  	return;

  bool on_left, on_right, on_up, on_down;
  on_left = on_right = on_up = on_down = false;

  if(x == 0)
  	on_left = true;
  if(x == width - 1)
  	on_right = true;
  if(y == 0)
  	on_up = true;
  if(y == height - 1)
  	on_down = true;

  __shared__ float neighbor_cost[4][DEPTH_NUM];
  __shared__ float neighbor_cost_min[4][DEPTH_NUM];
  __shared__ float raw_cost[4][DEPTH_NUM];

  neighbor_cost[dir][depth_id] = (data_devptr->atXY(x, y)).get_cost(depth_id);

  if(dir != 0 && !on_up) // to up
  {
  	neighbor_cost[dir][depth_id] += (dm_devptr->atXY(x, y - 1)).get_cost(depth_id);
  }
  if(dir != 1 && !on_down) // to down
  {
  	neighbor_cost[dir][depth_id] += (up_devptr->atXY(x, y + 1)).get_cost(depth_id);
  }
  if(dir != 2 && !on_left) // to left
  {
  	neighbor_cost[dir][depth_id] += (rm_devptr->atXY(x - 1, y)).get_cost(depth_id);
  }
  if(dir != 3 && !on_right) // to right
  {
  	neighbor_cost[dir][depth_id] += (lm_devptr->atXY(x + 1, y)).get_cost(depth_id);
  }
  neighbor_cost_min[dir][depth_id] = neighbor_cost[dir][depth_id];
  __syncthreads();

  //find min
  for(int i = DEPTH_NUM / 2; i > 0; i = i / 2)
  {
  	if(depth_id < i && neighbor_cost_min[dir][depth_id + i] < neighbor_cost_min[dir][depth_id])
  	{
  		neighbor_cost_min[dir][depth_id] = neighbor_cost_min[dir][depth_id + i];
  	}
  	__syncthreads();
  }

  //find min cost for every message
  float min_cost = neighbor_cost[dir][depth_id];
  if(depth_id > 0)
  	min_cost = fminf(min_cost, neighbor_cost[dir][depth_id - 1] + P1);
  if(depth_id < DEPTH_NUM - 1)
  	min_cost = fminf(min_cost, neighbor_cost[dir][depth_id + 1] + P1);
  min_cost = fminf(min_cost, neighbor_cost_min[dir][0] + P2);

  raw_cost[dir][depth_id] = min_cost;
  __syncthreads();

  for(int i = DEPTH_NUM / 2; i > 0; i = i / 2)
  {
  	if(depth_id < i)
  	{
  		raw_cost[dir][depth_id] += raw_cost[dir][depth_id + i];
  	}
  	__syncthreads();
  }

  min_cost = min_cost - raw_cost[dir][0] / (float) DEPTH_NUM;

  if(dir == 0) //up
  	(up_devptr->atXY(x,y)).set_cost(depth_id, min_cost);
  else if(dir == 1) //to down
  	(dm_devptr->atXY(x,y)).set_cost(depth_id, min_cost);
  else if(dir == 2) // to left
  	(lm_devptr->atXY(x,y)).set_cost(depth_id, min_cost);
  else // to right
  	(rm_devptr->atXY(x,y)).set_cost(depth_id, min_cost);
}

__global__ void depth_filter(	DeviceImage<PIXEL_COST> *data_devptr,
															DeviceImage<PIXEL_COST> *lm_devptr,
															DeviceImage<PIXEL_COST> *rm_devptr,
															DeviceImage<PIXEL_COST> *up_devptr,
															DeviceImage<PIXEL_COST> *dm_devptr,
															DeviceImage<float> *depth_devptr)
{
	const int x = blockIdx.x;
	const int y = blockIdx.y;
	const int depth_id = threadIdx.x;
	const int depth_id2 = depth_id + DEPTH_NUM / 2;
	const int width = data_devptr->width;
	const int height = data_devptr->height;

	__shared__ float cost[DEPTH_NUM];
	__shared__ float min_cost[DEPTH_NUM];
	__shared__ int cost_index[DEPTH_NUM];

	cost[depth_id] = (data_devptr->atXY(x,y)).get_cost(depth_id);
	cost[depth_id2] = (data_devptr->atXY(x,y)).get_cost(depth_id2);
	cost_index[depth_id] = depth_id;
	cost_index[depth_id2] = depth_id2;

	if(x + 1 < width) // add messgae from right
	{
		cost[depth_id] += (lm_devptr->atXY(x + 1,y)).get_cost(depth_id);
		cost[depth_id2] += (lm_devptr->atXY(x + 1,y)).get_cost(depth_id2);
	}
	if(x - 1 >= 0) // add messgae from left
	{
		cost[depth_id] += (rm_devptr->atXY(x - 1,y)).get_cost(depth_id);
		cost[depth_id2] += (rm_devptr->atXY(x - 1,y)).get_cost(depth_id2);
	}
	if(y + 1 < height) // add messgae from down
	{
		cost[depth_id] += (up_devptr->atXY(x,y + 1)).get_cost(depth_id);
		cost[depth_id2] += (up_devptr->atXY(x,y + 1)).get_cost(depth_id2);
	}
	if(y - 1 >= 0) // add messgae from up
	{
		cost[depth_id] += (dm_devptr->atXY(x,y - 1)).get_cost(depth_id);
		cost[depth_id2] += (dm_devptr->atXY(x,y - 1)).get_cost(depth_id2);
	}

	min_cost[depth_id] = cost[depth_id];
	min_cost[depth_id2] = cost[depth_id2];

	__syncthreads();
	for(int i = DEPTH_NUM / 2; i > 0; i = i / 2)
	{
		if(depth_id < i && min_cost[depth_id + i] < min_cost[depth_id])
		{
			min_cost[depth_id] = min_cost[depth_id + i];
			cost_index[depth_id] = cost_index[depth_id + i];
		}
		__syncthreads();
	}

	if(depth_id == 0)
	{
		if(cost_index[0] == 0 || cost_index[0] == DEPTH_NUM - 1)
			depth_devptr->atXY(x, y) = cost_index[0];
		else
		{
			float cost_pre = cost[cost_index[0] - 1];
			float cost_post = cost[cost_index[0] + 1];
      float a = cost_pre - 2.0f * min_cost[0] + cost_post;
      float b = - cost_pre + cost_post;
      depth_devptr->atXY(x,y) = (float) cost_index[0] - b / (2.0f * a);
		}
	}
}

__global__ void upsample(	DeviceImage<PIXEL_COST> *l1_message_devptr,
													DeviceImage<PIXEL_COST> *l0_message_devptr)
{
	const int depth_id = threadIdx.x;
	const int x = blockIdx.x; // in l1 image
	const int y = blockIdx.y; // in l1 image
	const int l0_width = l0_message_devptr->width;
	const int l0_height = l0_message_devptr->height;

	float value = (l1_message_devptr->atXY(x,y)).get_cost(depth_id);
	for(int i = 0; i < 2; i++)
	{
		for(int j = 0; j < 2; j++)
		{
			int x_up = x * 2 + i;
			int y_up = y * 2 + j;
			if(x_up < l0_width && y_up < l0_height)
				(l0_message_devptr->atXY(x_up, y_up)).set_cost(depth_id, value);
		}
	}
}
}