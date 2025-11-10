#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

// Define the number of threads per block
#define BLOCK_SIZE 1024  // Adjust this value based on your GPU's capabilities

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask,
                                    const int Batch, const int Map_out, const int Channel,
                                    const int Height, const int Width, const int K,
                                    const int Height_out, const int Width_out)
{
    // Macros for indexing the output and input tensors
    #define out_4d(i3, i2, i1, i0) output[( (size_t)(i3)*(Map_out*Height_out*Width_out) + \
                                           (i2)*(Height_out*Width_out) + \
                                           (i1)*(Width_out) + \
                                           (i0))]

    #define in_4d(i3, i2, i1, i0) input[( (size_t)(i3)*(Channel*Height*Width) + \
                                         (i2)*(Height*Width) + \
                                         (i1)*(Width) + \
                                         (i0))]

    #define mask_4d(i3, i2, i1, i0) mask[( (size_t)(i3)*(Channel*K*K) + \
                                            (i2)*(K*K) + \
                                            (i1)*K + \
                                            (i0))]

    // Compute the global thread index
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute the total number of output elements
    int total_elements = Batch * Map_out * Height_out * Width_out;

    if (index < total_elements) {
        // Compute the indices for the output element
        int w = index % Width_out;
        int h = (index / Width_out) % Height_out;
        int m = (index / (Width_out * Height_out)) % Map_out;
        int b = index / (Width_out * Height_out * Map_out);

        float acc = 0.0f;

        // Perform the convolution with loop unrolling hints
        // We apply #pragma unroll on the inner loops to reduce loop overhead.
        for (int c = 0; c < Channel; ++c) {
            #pragma unroll
            for (int p = 0; p < K; ++p) {
                #pragma unroll
                for (int q = 0; q < K; ++q) {
                    acc += in_4d(b, c, h + p, w + q) * mask_4d(m, c, p, q);
                }
            }
        }
        out_4d(b, m, h, w) = acc;
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
    // Allocate memory on the device
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    size_t input_size = (size_t)Batch * Channel * Height * Width * sizeof(float);
    size_t output_size = (size_t)Batch * Map_out * Height_out * Width_out * sizeof(float);
    size_t mask_size = (size_t)Map_out * Channel * K * K * sizeof(float);

    cudaMalloc((void**) device_input_ptr, input_size);
    cudaMalloc((void**) device_output_ptr, output_size);
    cudaMalloc((void**) device_mask_ptr, mask_size);

    // Copy data from host to device
    cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);

    // Error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cerr << "CUDA error (prolog): " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input,
                                             const float *device_mask,
                                             const int Batch, const int Map_out, const int Channel,
                                             const int Height, const int Width, const int K)
{
    // Compute output dimensions
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    // Compute the total number of output elements
    int total_elements = Batch * Map_out * Height_out * Width_out;

    // Set up execution configuration
    int threads_per_block = BLOCK_SIZE;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    // Launch the kernel with unrolled loops
    conv_forward_kernel<<<num_blocks, threads_per_block>>>(device_output, device_input, device_mask,
                                                           Batch, Map_out, Channel, Height, Width, K,
                                                           Height_out, Width_out);

    // Error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cerr << "CUDA error (kernel launch): " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output,
                                                    float *device_input, float *device_mask,
                                                    const int Batch, const int Map_out, const int Channel,
                                                    const int Height, const int Width, const int K)
{
    // Copy the output back to host
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    size_t output_size = (size_t)Batch * Map_out * Height_out * Width_out * sizeof(float);

    cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);

    // Error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cerr << "CUDA error (epilog): " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
}

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
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
