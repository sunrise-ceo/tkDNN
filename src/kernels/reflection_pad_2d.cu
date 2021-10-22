#include "kernels.h"
#include <thrust/pair.h>
#include <stdio.h>


__device__
inline thrust::pair<int64_t,int64_t> get_index_mapping2d(
        int64_t input_dim_x,int64_t input_dim_y,int64_t output_dim_x,
        int64_t output_dim_y,int64_t pad_l,int64_t pad_t,int64_t output_xy,
        int64_t y_shift,int64_t z_shift,int64_t n_plane){
    auto input_offset = ((blockIdx.y + y_shift) + (blockIdx.z + z_shift)*n_plane)*input_dim_x*input_dim_x;
    auto output_offset = ((blockIdx.y + y_shift) + (blockIdx.z + z_shift)*n_plane)*output_dim_x*output_dim_y;
    auto output_x = output_xy % output_dim_x;
    auto output_y = output_xy/output_dim_x;

    auto i_start_x = ::max(int64_t(0),-pad_l);
    auto i_start_y = ::max(int64_t(0),-pad_t);
    auto o_start_x = ::max(int64_t(0),pad_l);
    auto o_start_y = ::max(int64_t(0),pad_t);

    auto input_x = ::abs(output_x - pad_l) - ::abs(output_x - (input_dim_x + pad_l -1)) -output_x + 2*pad_l*input_dim_x -1 -o_start_x + i_start_x;
    auto input_y = ::abs(output_y -pad_t) - ::abs(output_y - (input_dim_y + pad_t -1) -output_y + 2*pad_t + input_dim_y -1 -o_start_y + i_start_y);

    return thrust::make_pair<int64_t,int64_t>(input_offset + input_y*input_dim_x + input_x,output_offset + output_y*output_dim_x+output_x);
}

__global__
void reflection_pad2d_out_kernel(
        float* input,float* output,int64_t input_dim_x,
        int64_t input_dim_y,int64_t pad_t,int64_t pad_b,int64_t pad_l,
        int64_t pad_r,int64_t y_shift,int64_t z_shift,int64_t n_plane){
    auto output_xy = threadIdx.x + blockIdx.x * blockDim.x;
    auto output_dim_x = input_dim_x + pad_l + pad_r;
    auto output_dim_y = input_dim_y + pad_t + pad_b;

    if(output_xy < output_dim_x*output_dim_y){
        auto index_pair = get_index_mapping2d(input_dim_x,input_dim_y,output_dim_x,output_dim_y,pad_l,pad_t,output_xy,y_shift,z_shift,n_plane);
        output[index_pair.second] = input[index_pair.first];
    }
}

int64_t ceilDiv(int64_t a,int64_t b){
    return (a+b-1)/b;
}


void reflection_pad2d_out_forward(int64_t padding[4],float *srcData,float *dstData,int64_t input_h,int64_t input_w,int64_t plane_dim,int64_t n_batch,cudaStream_t cudaStream){
    int64_t pad_l = padding[0];
    int64_t pad_r = padding[1];
    int64_t pad_t = padding[2];
    int64_t pad_b = padding[3];
    int64_t output_h = input_h + pad_t + pad_b;
    int64_t output_w = input_w + pad_l + pad_r;
    int64_t size_y = plane_dim;
    int64_t size_z = n_batch;
    int64_t output_plane_size = output_h*output_w;
    dim3 block_size(output_plane_size>256 ?256:output_plane_size);
    for(int64_t block_y=0;block_y<size_y;block_y += 65535){
        int64_t block_y_size = std::min(size_y - block_y,static_cast<int64_t>(65535));
        for(int64_t block_z=0;block_z<size_z;block_z += 65535){
            int64_t block_z_size = std::min(size_z -block_z,static_cast<int64_t>(65535));

            dim3 grid_size(ceilDiv(output_plane_size,static_cast<int64_t>(256)),block_y_size,block_z_size);
            reflection_pad2d_out_kernel<<<grid_size,block_size,0,cudaStream>>>(srcData,dstData,input_w,input_h,pad_t,pad_b,pad_l,pad_r,block_y,block_z,plane_dim);
        }
    }

}


