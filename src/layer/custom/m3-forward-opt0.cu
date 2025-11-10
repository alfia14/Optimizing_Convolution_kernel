#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define BLOCK_SIZE 1024
#define MAX_MASK_SIZE (1024*1024*4) // Adjust as needed

__constant__ float c_mask[MAX_MASK_SIZE];

__global__ void conv_forward_kernel_cm(float *output, const float *input,
                                       const float *mask_unused,
                                       int Batch, int Map_out, int Channel,
                                       int Height, int Width, int K,
                                       int Height_out, int Width_out)
{
    #define out_4d(i3, i2, i1, i0) output[((size_t)(i3)*(Map_out*Height_out*Width_out) + \
                                           (i2)*(Height_out*Width_out) + \
                                           (i1)*(Width_out) + \
                                           (i0))]
    #define in_4d(i3, i2, i1, i0) input[((size_t)(i3)*(Channel*Height*Width) + \
                                         (i2)*(Height*Width) + \
                                         (i1)*(Width) + \
                                         (i0))]

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = Batch * Map_out * Height_out * Width_out;

    if (index < total_elements) {
        int w = index % Width_out;
        int h = (index / Width_out) % Height_out;
        int m = (index / (Width_out * Height_out)) % Map_out;
        int b = index / (Width_out * Height_out * Map_out);

        int mask_base = m * (Channel*K*K);
        float acc = 0.0f;
        for (int c = 0; c < Channel; c++) {
            int c_base = mask_base + c*(K*K);
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                    acc += in_4d(b,c,h+p,w+q) * c_mask[c_base + p*K + q];
                }
            }
        }
        out_4d(b,m,h,w) = acc;
    }

    #undef out_4d
    #undef in_4d
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output,
                                                    const float *host_input,
                                                    const float *host_mask,
                                                    float **device_output_ptr,
                                                    float **device_input_ptr,
                                                    float **device_mask_ptr,
                                                    const int Batch,
                                                    const int Map_out,
                                                    const int Channel,
                                                    const int Height,
                                                    const int Width,
                                                    const int K)
{
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    size_t input_size = (size_t)Batch * Channel * Height * Width * sizeof(float);
    size_t output_size = (size_t)Batch * Map_out * Height_out * Width_out * sizeof(float);
    size_t mask_size = (size_t)Map_out * Channel * K * K * sizeof(float);

    cudaMalloc((void**)device_input_ptr, input_size);
    cudaMalloc((void**)device_output_ptr, output_size);
    cudaMalloc((void**)device_mask_ptr, mask_size); // allocated but unused

    cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);

    if ((mask_size / sizeof(float)) > MAX_MASK_SIZE) {
        std::cerr<<"Mask too large for constant memory."<<std::endl;
        exit(-1);
    }
    cudaMemcpyToSymbol(c_mask, host_mask, mask_size, 0, cudaMemcpyHostToDevice);

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cerr << "CUDA error (prolog): " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output,
                                             const float *device_input,
                                             const float *device_mask,
                                             const int Batch,
                                             const int Map_out,
                                             const int Channel,
                                             const int Height,
                                             const int Width,
                                             const int K)
{
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    int total_elements = Batch * Map_out * Height_out * Width_out;

    int threads_per_block = BLOCK_SIZE;
    int num_blocks = (total_elements + threads_per_block -1)/threads_per_block;

    conv_forward_kernel_cm<<<num_blocks, threads_per_block>>>(
        device_output, device_input, nullptr,
        Batch, Map_out, Channel, Height, Width, K, Height_out, Width_out);

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
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
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    size_t output_size = (size_t)Batch * Map_out * Height_out * Width_out * sizeof(float);

    cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cerr<<"CUDA error (epilog): "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
}

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; dev++)
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
