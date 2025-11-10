#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

// Adjust as needed for your GPU and problem sizes
#define TILE_WIDTH 16
#define BLOCK_SIZE 256

inline void checkCudaError(cudaError_t error, const char *msg) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error (" << msg << "): " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
}


__global__ void matrix_unrolling_kernel_1D(const float *input, float *output,
                                           int Batch, int Channel, int Height, int Width, int K)
{
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    if (Height_out <= 0 || Width_out <= 0) {
        return; // Invalid dimensions, no valid output
    }

    size_t Rows = (size_t)Channel * K * K;
    size_t Cols = (size_t)Batch * Height_out * Width_out;
    size_t total_elements = Rows * Cols;

    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int row = (int)(idx / Cols);
        int col = (int)(idx % Cols);

        int c = row / (K*K);
        int rem = row % (K*K);
        int p = rem / K;
        int q = rem % K;

        int b = (int)(col / (Height_out * Width_out));
        int temp = (int)(col % (Height_out * Width_out));
        int h = temp / Width_out;
        int w = temp % Width_out;

        #define in_4d(i3, i2, i1, i0) input[((size_t)(i3)*(Channel*Height*Width) + (i2)*(Height*Width) + (i1)*Width + (i0))]
        output[idx] = in_4d(b, c, h+p, w+q);
        #undef in_4d
    }
}


__global__ void matrixMultiplyShared(const float *A, const float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns)
{
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y, bx = blockIdx.x;
    int ty = threadIdx.y, tx = threadIdx.x;

    long long row = (long long)by * TILE_WIDTH + ty;
    long long col = (long long)bx * TILE_WIDTH + tx;

    float val = 0.0f;

    int tiles = (numAColumns + TILE_WIDTH - 1)/TILE_WIDTH;
    for (int tileId = 0; tileId < tiles; tileId++) {
        long long Acol = (long long)tileId*TILE_WIDTH + tx;
        long long Brow = (long long)tileId*TILE_WIDTH + ty;

        float aVal = 0.0f;
        float bVal = 0.0f;

        if (row < numARows && Acol < numAColumns) {
            aVal = A[row*(long long)numAColumns + Acol];
        }
        if (col < numCColumns && Brow < numBRows) {
            bVal = B[Brow*(long long)numBColumns + col];
        }

        tileA[ty][tx] = aVal;
        tileB[ty][tx] = bVal;

        __syncthreads();

        if (row < numCRows && col < numCColumns) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                val += tileA[ty][i] * tileB[i][tx];
            }
        }
        __syncthreads();
    }

    if (row < numCRows && col < numCColumns) {
        C[row*(long long)numCColumns + col] = val;
    }
}


__global__ void matrix_permute_kernel(const float *input, float *output, int Map_out,
                                      int Batch, int image_size) {
    int b = blockIdx.y;
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (x < image_size) {
        for (int m = 0; m < Map_out; m++) {
            size_t out_idx = ((size_t)b * Map_out * image_size) + m * image_size + x;
            size_t in_idx = ((size_t)m * Batch * image_size) + b * image_size + x;
            output[out_idx] = input[in_idx];
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input,
                                                    const float *host_mask,
                                                    float **device_output_ptr, float **device_input_ptr,
                                                    float **device_mask_ptr,
                                                    const int Batch, const int Map_out, const int Channel,
                                                    const int Height, const int Width, const int K)
{
    // Check for invalid dimensions upfront
    if (Height < K || Width < K) {
        std::cerr << "Invalid configuration: K cannot be greater than Height or Width." << std::endl;
        exit(-1);
    }

    cudaStream_t streamIn, streamCompute, streamOut;
    checkCudaError(cudaStreamCreate(&streamIn), "streamIn create");
    checkCudaError(cudaStreamCreate(&streamCompute), "streamCompute create");
    checkCudaError(cudaStreamCreate(&streamOut), "streamOut create");

    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;

    size_t input_size = (size_t)Batch * Channel * Height * Width * sizeof(float);
    size_t mask_size = (size_t)Map_out * Channel * K * K * sizeof(float);
    size_t output_size = (size_t)Batch * Map_out * Height_out * Width_out * sizeof(float);

    checkCudaError(cudaMalloc((void**)device_input_ptr, input_size), "malloc input");
    checkCudaError(cudaMalloc((void**)device_mask_ptr, mask_size), "malloc mask");
    checkCudaError(cudaMalloc((void**)device_output_ptr, output_size), "malloc output");

    checkCudaError(cudaMemcpyAsync(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice, streamIn), "H2D input");
    checkCudaError(cudaMemcpyAsync(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice, streamIn), "H2D mask");

    checkCudaError(cudaStreamSynchronize(streamIn), "streamIn sync");
    checkCudaError(cudaStreamDestroy(streamIn), "destroy streamIn");
    checkCudaError(cudaStreamDestroy(streamCompute), "destroy streamCompute");
    checkCudaError(cudaStreamDestroy(streamOut), "destroy streamOut");
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask,
                                             const int Batch, const int Map_out, const int Channel,
                                             const int Height, const int Width, const int K)
{
    // Recreate streams here
    cudaStream_t streamIn, streamCompute, streamOut;
    checkCudaError(cudaStreamCreate(&streamIn), "streamIn create");
    checkCudaError(cudaStreamCreate(&streamCompute), "streamCompute create");
    checkCudaError(cudaStreamCreate(&streamOut), "streamOut create");

    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;

    if (Height_out <= 0 || Width_out <= 0) {
        std::cerr << "Invalid output dimensions: Height_out=" << Height_out << ", Width_out=" << Width_out << std::endl;
        exit(-1);
    }

    // Compute unrolled dimensions
    size_t total_unrolled_elements = (size_t)Channel * (size_t)K * (size_t)K * (size_t)Batch * (size_t)Height_out * (size_t)Width_out;

    // Debug prints (uncomment if needed)
    // std::cout << "Batch=" << Batch << " Map_out=" << Map_out << " Channel=" << Channel << " Height=" << Height << " Width=" << Width << " K=" << K << std::endl;
    // std::cout << "Height_out=" << Height_out << " Width_out=" << Width_out << std::endl;
    // std::cout << "total_unrolled_elements=" << total_unrolled_elements << std::endl;

    float *unrolled_matrix;
    size_t unrolled_size = total_unrolled_elements * sizeof(float);
    checkCudaError(cudaMalloc((void**)&unrolled_matrix, unrolled_size), "malloc unrolled_matrix");

    // Unrolling kernel
    size_t threads = 1024;
    size_t blocks = (total_unrolled_elements + threads - 1)/threads;
    matrix_unrolling_kernel_1D<<<blocks, threads, 0, streamCompute>>>(device_input, unrolled_matrix,
                                                                      Batch, Channel, Height, Width, K);
    checkCudaError(cudaPeekAtLastError(), "unrolling kernel launch");
    checkCudaError(cudaStreamSynchronize(streamCompute), "unrolling kernel sync");

    // Dimensions for matmul
    int numARows = Map_out;
    int numAColumns = Channel * K * K;
    int numBRows = Channel * K * K;
    int numBColumns = Batch * Height_out * Width_out;
    int numCRows = Map_out;
    int numCColumns = Batch * Height_out * Width_out;

    // Check dimension consistency
    if (numAColumns != numBRows) {
        std::cerr << "Dimension mismatch: numAColumns=" << numAColumns << " numBRows=" << numBRows << std::endl;
        exit(-1);
    }

    float *matmul_output;
    size_t matmul_size = (size_t)numCRows * numCColumns * sizeof(float);
    checkCudaError(cudaMalloc((void**)&matmul_output, matmul_size), "malloc matmul_output");

    dim3 blockDimMatmul(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDimMatmul((numCColumns + TILE_WIDTH - 1)/TILE_WIDTH,
                       (numCRows + TILE_WIDTH - 1)/TILE_WIDTH);

    matrixMultiplyShared<<<gridDimMatmul, blockDimMatmul, 0, streamCompute>>>(
        device_mask, unrolled_matrix, matmul_output,
        numARows, numAColumns,
        numBRows, numBColumns,
        numCRows, numCColumns
    );
    checkCudaError(cudaPeekAtLastError(), "matmul kernel launch");
    checkCudaError(cudaStreamSynchronize(streamCompute), "matmul kernel sync");

    int out_image_size = Height_out * Width_out;
    dim3 permute_grid((out_image_size - 1) / BLOCK_SIZE + 1, Batch, 1);
    matrix_permute_kernel<<<permute_grid, BLOCK_SIZE, 0, streamCompute>>>(
        matmul_output, device_output, Map_out, Batch, out_image_size
    );
    checkCudaError(cudaPeekAtLastError(), "permute kernel launch");
    checkCudaError(cudaStreamSynchronize(streamCompute), "permute kernel sync");

    cudaFree(matmul_output);
    cudaFree(unrolled_matrix);

    checkCudaError(cudaStreamDestroy(streamIn), "destroy streamIn");
    checkCudaError(cudaStreamDestroy(streamCompute), "destroy streamCompute");
    checkCudaError(cudaStreamDestroy(streamOut), "destroy streamOut");
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output,
                                                    float *device_input, float *device_mask,
                                                    const int Batch, const int Map_out,
                                                    const int Channel, const int Height,
                                                    const int Width, const int K)
{
    cudaStream_t streamIn, streamCompute, streamOut;
    checkCudaError(cudaStreamCreate(&streamIn), "streamIn create");
    checkCudaError(cudaStreamCreate(&streamCompute), "streamCompute create");
    checkCudaError(cudaStreamCreate(&streamOut), "streamOut create");

    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    if (Height_out <= 0 || Width_out <= 0) {
        std::cerr << "Invalid output dimension in epilog" << std::endl;
        exit(-1);
    }

    size_t output_size = (size_t)Batch * Map_out * Height_out * Width_out * sizeof(float);

    checkCudaError(cudaMemcpyAsync(host_output, device_output, output_size, cudaMemcpyDeviceToHost, streamOut), "D2H output");
    checkCudaError(cudaStreamSynchronize(streamOut), "streamOut sync");

    cudaFree(device_input);
    cudaFree(device_mask);
    cudaFree(device_output);

    checkCudaError(cudaStreamDestroy(streamIn), "destroy streamIn");
    checkCudaError(cudaStreamDestroy(streamCompute), "destroy streamCompute");
    checkCudaError(cudaStreamDestroy(streamOut), "destroy streamOut");
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
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "
                 <<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "
                 <<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
