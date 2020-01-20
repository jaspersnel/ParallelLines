
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

__global__
void generate_hist(uint8_t* in, int* out, int rows, int cols) {
    extern __shared__ int hist[256 * 3];

    // Initialize empty hist
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        hist[i * 3] = 0;
        hist[(i * 3) + 1] = 0;
        hist[(i * 3) + 2] = 0;
    }

    // calculate hist
    for (int i = blockIdx.x + threadIdx.x * blockDim.x; i < rows * cols; i += gridDim.x * blockDim.x) {
        atomicAdd(&hist[in[i * 3] * 3], 1);
        atomicAdd(&hist[in[(i * 3) + 1] * 3 + 1], 1);
        atomicAdd(&hist[in[(i * 3) + 2] * 3 + 2], 1);
    }

    __syncthreads();

    // Write shared hists to global
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        out[blockIdx.x * 256 * 3 + i * 3] = hist[i * 3];
        out[blockIdx.x * 256 * 3 + i * 3 + 1] = hist[i * 3 + 1];
        out[blockIdx.x * 256 * 3 + i * 3 + 2] = hist[i * 3 + 2];
    }

    __syncthreads();

    // Have first block gather all hists in global
    if (blockIdx.x == 0) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            for (int j = 1; j < gridDim.x; j += 1) {
                out[i * 3] += out[j * 256 * 3 + i * 3];
                out[i * 3 + 1] += out[j * 256 * 3 + i * 3 + 1];
                out[i * 3 + 2] += out[j * 256 * 3 + i * 3 + 2];
            }
        }
    }

}

int main() {

    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    VideoCapture cap("C:/Users/Jasper/Videos/shelter2.mp4");

    // Check if camera opened successfully
    if (!cap.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    unsigned int num_blocks = 256;
    unsigned int num_threads = 256;

    // Set up in- and output arrays
    uint8_t* d_in;
    int* d_out;
    int h_out[256 * 3];

    bool init = true;
    while (1) {

        Mat frame;
        // Capture frame-by-frame
        cap >> frame;

        // For now, handle only 8-bit video
        CV_Assert(frame.depth() == CV_8U);

        // If the frame is empty, break immediately
        if (frame.empty())
            break;

        if (frame.isContinuous()) {
            // Do some init based on first frame
            if (init) {
                cudaMalloc(&d_in, frame.rows * frame.cols * sizeof(uchar) * frame.channels());
                cudaMalloc(&d_out, sizeof(int) * 256 * 3 * num_blocks);

                init = false;
            }

            // Copy frame to device and generate a hist
            cudaMemcpy(d_in, frame.data, frame.rows * frame.cols * frame.channels() * sizeof(uchar), cudaMemcpyHostToDevice);
            generate_hist << <num_blocks, num_threads >> > (d_in, d_out, frame.rows, frame.cols);

            cudaDeviceSynchronize();

            // Copy back to host and print
            // TODO: do something useful with this lol
            cudaMemcpy(h_out, d_out, 256 * 3 * sizeof(int), cudaMemcpyDeviceToHost);
            for (int i = 0; i < 256 * 3; i++) {
                printf("%d,", h_out[i]);
            }
            printf("\n");

        }
        else {
            cout << "Frame is not continuous" << endl;
        }

        // Press  ESC on keyboard to exit
        //char c = (char)waitKey(25);
        //if (c == 27)
        //    break;
    }

    // When everything done, release the video capture object
    cap.release();

    // Free cuda memory
    cudaFree(d_in);
    cudaFree(d_out);

    // Closes all the frames
    destroyAllWindows();

    return 0;
}