
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <chrono>
#include <ctime>

#define NUM_OF_RANDOM_RESTARTS 40

using namespace cv;
using namespace std;

struct centroid {
	int currR;
	int currG;
	int currB;

	int currNumberOfPointsEnclosed;

};

cudaError_t kmeanClustering(int* labels, float* distances, int* temp_host_centroid_R, int* temp_host_centroid_G, int* temp_host_centroid_B, int* Rarray, int* Garray, int* Barray, int k, unsigned int block_per_grid, unsigned int thread_per_block);

__global__ void kmeanKernel(int* labels, float* distances, int* centroids_Rarray, int* centroids_Garray, int* centroids_Barray, int* R, int* G, int* B, int* k, int* image_width)
{
	int currIndx = threadIdx.x*(*image_width) + blockIdx.x;
	//printf("Hello from block %d, thread %d\n\t My pixel has the values R:%d G:%d B:%d\n\n", blockIdx.x, threadIdx.x, R[currIndx], G[currIndx], B[currIndx]);

	float shortestEuclideanDistance = 100000; //some fluff
	int labelToClosestCentroid = -1;
	//find nearest centroid using Euclidean distance
	for (int i = 0; i < *k; i++)
	{
		float distance = sqrtf(powf((centroids_Rarray[i] - R[currIndx]), 2) + powf((centroids_Garray[i] - G[currIndx]), 2) + powf((centroids_Barray[i] - B[currIndx]), 2));

		if (distance < shortestEuclideanDistance)
		{
			shortestEuclideanDistance = distance;
			labelToClosestCentroid = i;
		}
	}

	//return nearest centroid information
	labels[currIndx] = labelToClosestCentroid;
	distances[currIndx] = shortestEuclideanDistance;
}

int main(int argc, char** argv)
{
	srand(time(0));

	if (argc != 3)
	{
		cout << " Usage: kMean_ImageSeg_CUDA kValue ImageToLoadAndDisplay" << endl;
		return -1;
	}

	//======grab k value======
	int k = stoi(argv[1]);

	//======get your immage======
	Mat image;
	image = imread(argv[2], IMREAD_COLOR); // Read the file
	if (image.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	else {
		printf("Current subject image is %d x %d.\n", image.rows, image.cols);
	}

	namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window", image); // Show our image inside it.

	waitKey(0); // Wait for a keystroke in the window

	//initialize variables in order to keep track of the best clustering across the
		//random restarts
	float best_clusterRadiusmean = 10000; //will store the average radius of the clusters for the best run
		//average cluster radius is a good predictor of how good your clustering is. the lower the better.
	Mat best_image; //will store the image of the best run
	
	std::chrono::duration<double> totalSeconds; //used to acquire average run time

	int currNumOfRestarts = 0;
	while (currNumOfRestarts < NUM_OF_RANDOM_RESTARTS)
	{
		printf("\n===============\nStarting run %d\n===============\n", currNumOfRestarts+1);
		centroid* centroidList = new centroid[k];

		//======centroid seeding======
		for (int i = 0; i < k; i++)
		{
			int randomRow = rand() % image.rows;
			int randomCol = rand() % image.cols;

			printf("Choosing random centroid start at pixel %d,%d.\n", randomRow, randomCol);

			centroidList[i].currR = image.at<cv::Vec3b>(randomRow, randomCol)[2];
			centroidList[i].currG = image.at<cv::Vec3b>(randomRow, randomCol)[1];
			centroidList[i].currB = image.at<cv::Vec3b>(randomRow, randomCol)[0];
			centroidList[i].currNumberOfPointsEnclosed = 1;
		}

		//array of just the RGB valuse of each pixel
		int* rawRarray = new int[image.rows*image.cols];
		int* rawGarray = new int[image.rows*image.cols];
		int* rawBarray = new int[image.rows*image.cols];

		for (int i = 0; i < image.cols; i++)
		{
			for (int j = 0; j < image.rows; j++)
			{
				rawBarray[i*image.rows + j] = image.at<cv::Vec3b>(j, i)[0]; //B
				rawGarray[i*image.rows + j] = image.at<cv::Vec3b>(j, i)[1]; //G
				rawRarray[i*image.rows + j] = image.at<cv::Vec3b>(j, i)[2]; //R
			}
		}


		//set up labels array to use in tracking changes in the standard label array during execution
		int* master_labels = new int[image.rows*image.cols];

		for (int i = 0; i < image.cols; i++)
		{
			for (int j = 0; j < image.rows; j++)
			{
				master_labels[i*image.rows + j] = -1;
			}
		}


		int numOfPasses = 0;
		bool converged = false;

		auto start = std::chrono::system_clock::now();
		while (!converged) {
			numOfPasses++;
			//======initialize labeling matrix======
			int* labels = new int[image.rows*image.cols];
			//initialize an array to store the distance of point to their labeled centroid's center. This makes comparison better random restarts easier
			float* distances = new float[image.rows*image.cols];

			for (int i = 0; i < image.cols; i++)
			{
				for (int j = 0; j < image.rows; j++)
				{
					labels[i*image.rows + j] = -1;
					distances[i*image.rows + j] = 1000;
				}
			}

			int blocksize = image.rows;//number of threads per block
			int gridsize = image.cols;//number of blocks per grid

			//load RGB values of centroid into dynbamically allocated arrays to make it easier to pass off to device memory
			int* temp_host_centroid_R = new int[k];
			int* temp_host_centroid_G = new int[k];
			int* temp_host_centroid_B = new int[k];

			for (int i = 0; i < k; i++)
			{
				temp_host_centroid_R[i] = centroidList[i].currR;
				temp_host_centroid_G[i] = centroidList[i].currG;
				temp_host_centroid_B[i] = centroidList[i].currB;
			}


			cudaError_t kmeanCudaStatus = kmeanClustering(labels, distances, temp_host_centroid_R, temp_host_centroid_G, temp_host_centroid_B, rawRarray, rawGarray, rawBarray, k, gridsize, blocksize);
			if (kmeanCudaStatus != cudaSuccess) {
				fprintf(stderr, "kmeanClustering failed!");
				return 1;
			}

			//waitKey(0); // Wait for a keystroke in the window

			//======POST PROCESSES LABELS======
			//flush centroid means
			for (int i = 0; i < k; i++)
			{
				centroidList[i].currR = 0;
				centroidList[i].currG = 0;
				centroidList[i].currB = 0;
				centroidList[i].currNumberOfPointsEnclosed = 0;
			}
			//assign new centroid total values (just totaling everything up to take the mean right after)
			for (int i = 0; i < image.cols; i++)
			{
				for (int j = 0; j < image.rows; j++)
				{
					centroidList[labels[i*image.rows + j]].currR = centroidList[labels[i*image.rows + j]].currR + rawRarray[i*image.rows + j];
					centroidList[labels[i*image.rows + j]].currG = centroidList[labels[i*image.rows + j]].currG + rawGarray[i*image.rows + j];
					centroidList[labels[i*image.rows + j]].currB = centroidList[labels[i*image.rows + j]].currB + rawBarray[i*image.rows + j];
					centroidList[labels[i*image.rows + j]].currNumberOfPointsEnclosed++;
				}
			}
			//calculate the new mean (or centroid center)
			for (int i = 0; i < k; i++)
			{
				if (centroidList[i].currNumberOfPointsEnclosed != 0)
				{
					centroidList[i].currR = centroidList[i].currR / centroidList[i].currNumberOfPointsEnclosed;
					centroidList[i].currG = centroidList[i].currG / centroidList[i].currNumberOfPointsEnclosed;
					centroidList[i].currB = centroidList[i].currB / centroidList[i].currNumberOfPointsEnclosed;
				}
				else {
					//fprintf(stderr, "... A cluster doesn't include its starting node?!");
					//waitKey(0); // Wait for a keystroke in the window
				}
			}

			//======PRINT CURRENT STATE IMAGE=====
			Mat imageCopy = image.clone();

			//store the effects of the image segmenting into the image copy
			for (int i = 0; i < imageCopy.cols; i++)
			{
				for (int j = 0; j < imageCopy.rows; j++)
				{
					imageCopy.at<cv::Vec3b>(j, i)[0] = centroidList[labels[i*imageCopy.rows + j]].currB; //B
					imageCopy.at<cv::Vec3b>(j, i)[1] = centroidList[labels[i*imageCopy.rows + j]].currG; //G
					imageCopy.at<cv::Vec3b>(j, i)[2] = centroidList[labels[i*imageCopy.rows + j]].currR; //R
				}
			}

			//char nameBuffer[50];
			//sprintf(nameBuffer, "Progress at iteration %d", numOfPasses);

			namedWindow("Current Progress State", WINDOW_AUTOSIZE); // Create a window for display.
			imshow("Current Progress State", imageCopy); // Show our image inside it.

			//check for convergence
			converged = true;
			int numberReassigned = 0;
			for (int i = 0; i < image.cols; i++)
			{
				for (int j = 0; j < image.rows; j++)
				{
					if (master_labels[i*image.rows + j] != labels[i*image.rows + j])
					{
						converged = false;
						master_labels[i*image.rows + j] = labels[i*image.rows + j]; //record for next convergence check
						numberReassigned++;
					}
				}
			}
			printf("\n==========\nOn pass %d, %d labels were reassigned.\n==========\n", numOfPasses, numberReassigned);
			//	waitKey(0); // Wait for a keystroke in the window

			//if converged, see how this run compared to the previous random restarts
			if (converged)
			{
				float* centroidRadius = new float[k];
				for (int i = 0; i < k; i++)
				{
					centroidRadius[i] = 0;
				}

				for (int i = 0; i < image.cols; i++)
				{
					for (int j = 0; j < image.rows; j++)
					{
						if (distances[i*image.rows + j] > centroidRadius[labels[i*image.rows + j]])
						{
							centroidRadius[labels[i*image.rows + j]] = distances[i*image.rows + j];
						}
					}
				}

				float curr_centroidRadiusMean = 0;
				for (int i = 0; i < k; i++)
				{
					curr_centroidRadiusMean = curr_centroidRadiusMean + centroidRadius[i];
				}
				curr_centroidRadiusMean = curr_centroidRadiusMean / k;

				if (best_clusterRadiusmean > curr_centroidRadiusMean)
				{
					best_clusterRadiusmean = curr_centroidRadiusMean;
					best_image = imageCopy.clone();
				}

			}

			delete labels;
			delete temp_host_centroid_R;
			delete temp_host_centroid_G;
			delete temp_host_centroid_B;
		}
		//gather our timing information
		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;
		std::time_t end_time = std::chrono::system_clock::to_time_t(end);

		if (currNumOfRestarts == 0)
		{
			totalSeconds = elapsed_seconds;
		}
		else
		{
			totalSeconds = totalSeconds + elapsed_seconds;
		}

		currNumOfRestarts++;
	}

	std::cout << "average elapsed time: " << totalSeconds.count()/ NUM_OF_RANDOM_RESTARTS << "s\n";


	namedWindow("Best Image Segmentation Among Restarts", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Best Image Segmentation Among Restarts", best_image); // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window

	//save the segmented image
	char nameBuffer[100];
	string temp = argv[2];
	string extension = temp.substr(temp.find_last_of("."));
	string pictureFileWithoutExtension = temp.substr(0, temp.find_last_of("."));
	sprintf(nameBuffer, "%s-result-k%d-%drestartsAllowed%s", pictureFileWithoutExtension.c_str(), k, NUM_OF_RANDOM_RESTARTS, extension.c_str());
	imwrite(nameBuffer, best_image);

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t kmeanClustering(int* labels, float* distances, int* temp_host_centroid_R, int* temp_host_centroid_G, int* temp_host_centroid_B, int* Rarray, int* Garray, int* Barray, int k, unsigned int block_per_grid, unsigned int thread_per_block)
{
	//initialize device allocations
	int* dev_R;
	int* dev_G;
	int* dev_B;
	int* dev_size;
	int* dev_label;
	float* dev_distances;
	int* dev_k;
	int* dev_centroid_R;
	int* dev_centroid_G;
	int* dev_centroid_B;


	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	
	// Allocate GPU buffer for device based variables
	cudaStatus = cudaMalloc((void**)&dev_R, block_per_grid*thread_per_block * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_G, block_per_grid*thread_per_block * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_B, block_per_grid*thread_per_block * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_size, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_label, block_per_grid*thread_per_block * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_distances, block_per_grid*thread_per_block * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_k, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_centroid_R, block_per_grid*thread_per_block * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_centroid_G, block_per_grid*thread_per_block * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_centroid_B, block_per_grid*thread_per_block * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_R, Rarray, block_per_grid*thread_per_block*sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_G, Garray, block_per_grid*thread_per_block * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_B, Barray, block_per_grid*thread_per_block * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_size, &block_per_grid, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_label, labels, block_per_grid*thread_per_block * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_distances, distances, block_per_grid*thread_per_block * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_k, &k, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_centroid_R, temp_host_centroid_R, k * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_centroid_G, temp_host_centroid_G, k * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_centroid_B, temp_host_centroid_B, k * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	kmeanKernel << <block_per_grid, thread_per_block >> >(dev_label, dev_distances, dev_centroid_R, dev_centroid_G, dev_centroid_B, dev_R,dev_G,dev_B, dev_k, dev_size);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(labels, dev_label, block_per_grid*thread_per_block * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(distances, dev_distances, block_per_grid*thread_per_block * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	//Clean up our allocated variables *clean up... clean up... everybody do your share*
	cudaFree(dev_R);
	cudaFree(dev_G);
	cudaFree(dev_B);
	cudaFree(dev_centroid_R);
	cudaFree(dev_centroid_G);
	cudaFree(dev_centroid_B);
	cudaFree(dev_k);
	cudaFree(dev_label);
	cudaFree(dev_distances);
	cudaFree(dev_size);


	return cudaStatus;
}
