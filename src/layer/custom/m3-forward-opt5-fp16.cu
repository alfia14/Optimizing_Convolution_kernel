#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cuda_fp16.h> // for __half

#define BLOCK_SIZE 1024

// Kernel to convert float array to half array
__global__ void float_to_half_kernel(const float *src, __half *dst, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
    }
}

// Kernel to convert half array to float array
__global__ void half_to_float_kernel(const __half *src, float *dst, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __half2float(src[idx]);
    }
}

__global__ void conv_forward_kernel(__half *output, const __half *input, const __half *mask,
                                    int Batch, int Map_out, int Channel,
                                    int Height, int Width, int K,
                                    int Height_out, int Width_out)
{
    // Macros for indexing
    #define out_4d(i3, i2, i1, i0) output[((size_t)(i3)*(Map_out*Height_out*Width_out) + \
                                           (i2)*(Height_out*Width_out) + \
                                           (i1)*(Width_out) + \
                                           (i0))]

    #define in_4d(i3, i2, i1, i0) input[((size_t)(i3)*(Channel*Height*Width) + \
                                         (i2)*(Height*Width) + \
                                         (i1)*Width + \
                                         (i0))]

    #define mask_4d(i3, i2, i1, i0) mask[((size_t)(i3)*(Channel*K*K) + \
                                          (i2)*(K*K) + \
                                          (i1)*K + \
                                          (i0))]

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = Batch * Map_out * Height_out * Width_out;

    if (index < total_elements) {
        int w = index % Width_out;
        int h = (index / Width_out) % Height_out;
        int m = (index / (Width_out * Height_out)) % Map_out;
        int b = index / (Width_out * Height_out * Map_out);

        // Accumulate in float to reduce precision loss
        float acc = 0.0f;

        for (int c = 0; c < Channel; c++) {
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                    // Convert halves to floats
                    float in_val = 0.0f;
                    float mask_val = 0.0f;
                    if ((b < Batch) && (c < Channel) && (h+p < Height) && (w+q < Width))
                        in_val = __half2float(in_4d(b,c,h+p,w+q));
                    mask_val = __half2float(mask_4d(m,c,p,q));
                    acc += in_val * mask_val;
                }
            }
        }

        // Convert acc back to half
        out_4d(b,m,h,w) = __float2half(acc);
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input,
                                                    const float *host_mask,
                                                    float **device_output_ptr, float **device_input_ptr,
                                                    float **device_mask_ptr,
                                                    const int Batch, const int Map_out, const int Channel,
                                                    const int Height, const int Width, const int K)
{
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;

    size_t input_size = (size_t)Batch * Channel * Height * Width;
    size_t output_size = (size_t)Batch * Map_out * Height_out * Width_out;
    size_t mask_size = (size_t)Map_out * Channel * K * K;

    // Allocate float arrays as per original M2 code for input/mask
    float *device_input_float, *device_mask_float, *device_output_float;
    cudaMalloc((void**)&device_input_float, input_size*sizeof(float));
    cudaMalloc((void**)&device_output_float, output_size*sizeof(float));
    cudaMalloc((void**)&device_mask_float, mask_size*sizeof(float));

    cudaMemcpy(device_input_float, host_input, input_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_mask_float, host_mask, mask_size*sizeof(float), cudaMemcpyHostToDevice);

    // Allocate half arrays
    __half *device_input_half, *device_mask_half, *device_output_half;
    cudaMalloc((void**)&device_input_half, input_size*sizeof(__half));
    cudaMalloc((void**)&device_mask_half, mask_size*sizeof(__half));
    cudaMalloc((void**)&device_output_half, output_size*sizeof(__half));

    // Convert input float->half
    {
        size_t total = input_size;
        int threads=BLOCK_SIZE;
        int blocks=(int)((total+threads-1)/threads);
        float_to_half_kernel<<<blocks,threads>>>(device_input_float, device_input_half, total);
        cudaDeviceSynchronize();
    }

    // Convert mask float->half
    {
        size_t total=mask_size;
        int threads=BLOCK_SIZE;
        int blocks=(int)((total+threads-1)/threads);
        float_to_half_kernel<<<blocks,threads>>>(device_mask_float, device_mask_half, total);
        cudaDeviceSynchronize();
    }

    // We don't need to convert output yet (it will be computed in half)
    // Free temporary float arrays for input and mask
    cudaFree(device_input_float);
    cudaFree(device_mask_float);

    // Assign back pointers
    *device_input_ptr = (float*)device_input_half;   // careful cast
    *device_mask_ptr = (float*)device_mask_half;
    *device_output_ptr = (float*)device_output_half; // store as float* but really half*

    // Store output_float pointer for epilog usage in cast?
    // We will allocate output float arrays in epilog for final conversion.

    cudaError_t error=cudaGetLastError();
    if(error!=cudaSuccess){
        std::cerr<<"CUDA error (prolog): "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output,
                                             const float *device_input,
                                             const float *device_mask,
                                             const int Batch, const int Map_out,
                                             const int Channel,
                                             const int Height,
                                             const int Width,
                                             const int K)
{
    int Height_out=Height-K+1;
    int Width_out=Width-K+1;
    int total_elements=Batch*Map_out*Height_out*Width_out;

    int threads_per_block=BLOCK_SIZE;
    int num_blocks=(total_elements+threads_per_block-1)/threads_per_block;

    __half *d_output_half = (__half*)device_output;
    const __half *d_input_half = (const __half*)device_input;
    const __half *d_mask_half = (const __half*)device_mask;

    conv_forward_kernel<<<num_blocks, threads_per_block>>>(d_output_half, d_input_half, d_mask_half,
                                                           Batch, Map_out, Channel,
                                                           Height, Width, K,
                                                           Height_out, Width_out);
    cudaError_t error=cudaGetLastError();
    if(error!=cudaSuccess){
        std::cerr<<"CUDA error (kernel launch): "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output,
                                                    float *device_output,
                                                    float *device_input,
                                                    float *device_mask,
                                                    const int Batch,
                                                    const int Map_out,
                                                    const int Channel,
                                                    const int Height,
                                                    const int Width,
                                                    const int K)
{
    int Height_out=Height-K+1;
    int Width_out=Width-K+1;
    size_t output_size=(size_t)Batch*Map_out*Height_out*Width_out;

    // device_output is in half
    __half *d_output_half = (__half*)device_output;

    // Allocate a float buffer to convert half->float
    float *d_output_float;
    cudaMalloc((void**)&d_output_float, output_size*sizeof(float));

    {
        size_t total=output_size;
        int threads=BLOCK_SIZE;
        int blocks=(int)((total+threads-1)/threads);
        half_to_float_kernel<<<blocks,threads>>>(d_output_half, d_output_float, total);
        cudaDeviceSynchronize();
    }

    // copy back to host
    cudaMemcpy(host_output, d_output_float, output_size*sizeof(float), cudaMemcpyDeviceToHost);

    // Free all device memory
    cudaFree(d_output_float);
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);

    cudaError_t error=cudaGetLastError();
    if(error!=cudaSuccess){
        std::cerr<<"CUDA error (epilog): "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
}

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for(int dev=0; dev<deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<
                  deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<
                  deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
