#include <cstdio>
#include <algorithm>
#include <cstring>
#include "kernels.h"
#include <errno.h>

#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}


__device__ float dmcn_im2col_bilinear(const float *bottom_data, const int data_width,
                                      const int height, const int width, float h, float w)
{
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  float lh = h - h_low;
  float lw = w - w_low;
  float hh = 1 - lh, hw = 1 - lw;

  float v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = bottom_data[h_low * data_width + w_low];
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = bottom_data[h_low * data_width + w_high];
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = bottom_data[h_high * data_width + w_low];
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom_data[h_high * data_width + w_high];

  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

__global__ void modulated_deformable_im2col_gpu_kernel(const int n,
                                                       const float *data_im, const float *data_offset, const float *data_mask,
                                                       const int height, const int width, const int kernel_h, const int kernel_w,
                                                       const int pad_h, const int pad_w,
                                                       const int stride_h, const int stride_w,
                                                       const int dilation_h, const int dilation_w,
                                                       const int channel_per_deformable_group,
                                                       const int batch_size, const int num_channels, const int deformable_group,
                                                       const int height_col, const int width_col,
                                                       float *data_col)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int b_col = (index / width_col / height_col) % batch_size;
    const int c_im = (index / width_col / height_col) / batch_size;
    const int c_col = c_im * kernel_h * kernel_w;

    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;

    float *data_col_ptr = data_col + ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    //const float* data_im_ptr = data_im + ((b_col * num_channels + c_im) * height + h_in) * width + w_in;
    const float *data_im_ptr = data_im + (b_col * num_channels + c_im) * height * width;
    const float *data_offset_ptr = data_offset + (b_col * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;

    const float *data_mask_ptr = data_mask + (b_col * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;

    for (int i = 0; i < kernel_h; ++i)
    {
      for (int j = 0; j < kernel_w; ++j)
      {
        const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
        const int data_mask_hw_ptr = ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;
        const float offset_h = data_offset_ptr[data_offset_h_ptr];
        const float offset_w = data_offset_ptr[data_offset_w_ptr];
        const float mask = data_mask_ptr[data_mask_hw_ptr];
        float val = static_cast<float>(0);
        const float h_im = h_in + i * dilation_h + offset_h;
        const float w_im = w_in + j * dilation_w + offset_w;
        //if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width)
        {
          //const float map_h = i * dilation_h + offset_h;
          //const float map_w = j * dilation_w + offset_w;
          //const int cur_height = height - h_in;
          //const int cur_width = width - w_in;
          //val = dmcn_im2col_bilinear(data_im_ptr, width, cur_height, cur_width, map_h, map_w);
          val = dmcn_im2col_bilinear(data_im_ptr, width, height, width, h_im, w_im);
        }
        *data_col_ptr = val * mask;
        data_col_ptr += batch_size * height_col * width_col;
        //data_col_ptr += height_col * width_col;
      }
    }
  }
}


void modulated_deformable_im2col_cuda(cudaStream_t stream,
  const float* data_im, const float* data_offset, const float* data_mask,
  const int batch_size, const int channels, const int height_im, const int width_im, 
  const int height_col, const int width_col, const int kernel_h, const int kenerl_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
  const int dilation_h, const int dilation_w,
  const int deformable_group, float* data_col) {
  // num_axes should be smaller than block size
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * batch_size * height_col * width_col;
  modulated_deformable_im2col_gpu_kernel
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
          0, stream>>>(
      num_kernels, data_im, data_offset, data_mask, height_im, width_im, kernel_h, kenerl_w,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, channel_per_deformable_group,
      batch_size, channels, deformable_group, height_col, width_col, data_col);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in modulated_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }

}


void dcn_v2_cuda_forward(float *input, float *weight,
                         float *bias, float *ones,
                         float *offset, float *mask,
                         float *output, float *columns,
                         int kernel_h, int kernel_w,
                         const int stride_h, const int stride_w,
                         const int pad_h, const int pad_w,
                         const int dilation_h, const int dilation_w,
                         const int deformable_group,
                         const int in_n, const int in_c, const int in_h, const int in_w, 
                         const int out_n, const int out_c, const int out_h, const int out_w,
                         const int dst_dim, cudaStream_t stream)
{
  checkCuda(cudaDeviceSynchronize());
  cudaError_t cudaStat;    
  cublasStatus_t stat;
  cublasHandle_t handle;
  stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("CUBLAS initialization failed\n");
      return;
  }
  checkCuda(cudaDeviceSynchronize());
  
  const int batch = in_n;
  const int channels = in_c;
  const int height = in_h;
  const int width = in_w;

  const int channels_out = out_c;
  const int channels_kernel = in_c;
  const int kernel_h_ = kernel_h;
  const int kernel_w_ = kernel_w;

  const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  float *input_n;
  cudaMalloc(&input_n, (in_n*in_c*in_h*in_w)*sizeof(float));
  cudaMemcpy(input_n, input, (in_n*in_c*in_h*in_w)*sizeof(float), cudaMemcpyDeviceToDevice);
  checkCuda(cudaDeviceSynchronize());

  float *offset_n;
  cudaMalloc(&offset_n, ((dst_dim/3)*2)*sizeof(float));
  cudaMemcpy(offset_n, offset, ((dst_dim/3)*2)*sizeof(float), cudaMemcpyDeviceToDevice);
  checkCuda(cudaDeviceSynchronize());

  float *mask_n;
  cudaMalloc(&mask_n, (dst_dim/3)*sizeof(float));
  cudaMemcpy(mask_n, mask, (dst_dim/3)*sizeof(float), cudaMemcpyDeviceToDevice);
  checkCuda(cudaDeviceSynchronize());

  float *output_n;
  checkCuda(cudaMalloc(&output_n, (channels_out*height_out*width_out)*sizeof(float)));
  checkCuda(cudaDeviceSynchronize());
  
  long m_ = channels_out;
  long n_ = height_out * width_out;
  long k_ = 1;
  float alpha = 1.0;
  float beta = 0.0;
  checkCuda(cudaDeviceSynchronize());
  stat = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
              n_, m_, k_, &alpha, 
              ones, k_, bias, k_, 
              &beta, output_n, n_);
  if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("CUBLAS initialization failed\n");
      return ;
  }

  checkCuda(cudaDeviceSynchronize());

  modulated_deformable_im2col_cuda(stream,
                                    input_n, offset_n,
                                    mask_n,
                                    1, channels, height, width,
                                    height_out, width_out, kernel_h, kernel_w,
                                    pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                                    deformable_group, columns);
  checkCuda(cudaDeviceSynchronize());

  //(k * m)  x  (m * n)
  // Y = WC
  long m = channels_out;
  long n = height_out * width_out;
  long k = channels * kernel_h * kernel_w;

  alpha = 1.0;
  beta = 1.0;
  
  stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
              n, m, k, &alpha, 
              columns, n, weight, k, 
              &beta, output_n, n);

  cudaMemcpy(output, output_n, (n*m)*sizeof(float), cudaMemcpyDeviceToDevice);  
  checkCuda(cudaDeviceSynchronize());
            
  if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("CUBLAS initialization failed\n");
      return ;
  }
}
