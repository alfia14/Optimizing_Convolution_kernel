#include "cpu-new-forward.h"

void conv_forward_cpu(float *output, const float *input, const float *mask,
                      const int Batch, const int Map_out, const int Channel,
                      const int Height, const int Width, const int K)
{
    /*
    Implement the forward pass for the convolutional layer.
    The goal here is to be correct, not fast (this is the CPU implementation.)

    Function parameters:
    output   - Output array
    input    - Input array
    mask     - Convolution kernel (weights)
    Batch    - Batch size (number of images)
    Map_out  - Number of output feature maps
    Channel  - Number of input feature maps (channels)
    Height   - Input height dimension
    Width    - Input width dimension
    K        - Kernel height and width (assumed square)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    // Macros for indexing the arrays
    #define out_4d(b, m, h, w) output[(b)*(Map_out*Height_out*Width_out) + \
                                        (m)*(Height_out*Width_out) + \
                                        (h)*(Width_out) + \
                                        (w)]
    #define in_4d(b, c, h, w) input[(b)*(Channel*Height*Width) + \
                                     (c)*(Height*Width) + \
                                     (h)*(Width) + \
                                     (w)]
    #define mask_4d(m, c, p, q) mask[(m)*(Channel*K*K) + \
                                      (c)*(K*K) + \
                                      (p)*(K) + \
                                      (q)]

    // Implement the convolution operation over the batch
    // For each image in the batch
    for (int b = 0; b < Batch; ++b) {
        // For each output feature map (output channel)
        for (int m = 0; m < Map_out; ++m) {
            // For each output pixel (spatial dimensions)
            for (int h = 0; h < Height_out; ++h) {
                for (int w = 0; w < Width_out; ++w) {
                    // Initialize the output value to zero
                    out_4d(b, m, h, w) = 0.0f;
                    // Sum over all input channels and kernel positions
                    for (int c = 0; c < Channel; ++c) {
                        for (int p = 0; p < K; ++p) {
                            for (int q = 0; q < K; ++q) {
                                // Compute the convolution
                                out_4d(b, m, h, w) +=
                                    in_4d(b, c, h + p, w + q) * mask_4d(m, c, p, q);
                            }
                        }
                    }
                }
            }
        }
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}
