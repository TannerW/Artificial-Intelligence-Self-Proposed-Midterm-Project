# Artificial-Intelligence-Self-Proposed-Midterm-Project
# Author: Owen Tanner Wilkerson

Purpose:
To perform image segmentation through k-means that has been parallelized across a GPU.

Testing results directory:
./testImages/

Source code file:
./kMean_ImageSeg_CUDA/kMean_ImageSeg_CUDA/kernel.cu

How to run the code:
This code takes a subsantial amount of setup.
1) you will need a CUDA compatible Nvidia GPU installed on the machine you plan to run this code on.
2) you will need to install Visual Studio 2015
3) install CUDA9.1 for Visual Studio 2015
	install guide: https://developer.download.nvidia.com/compute/cuda/9.1/Prod/docs/sidebar/CUDA_Installation_Guide_Windows.pdf

4) install OpenCV3.4 for Visual Studio 2015
	install guide: https://docs.opencv.org/3.4/d3/d52/tutorial_windows_install.html and https://docs.opencv.org/3.4/dd/d6e/tutorial_windows_visual_studio_opencv.html

5) then, after 3 and 4 are setup properly, the code can be run in Visual Studio under the follow command line argument structure Command Arguments: [k-value] [path to target image]


Controlling random restarts:
The number of random restarts is simply defined at the top of the source code ./kMean_ImageSeg_CUDA/kMean_ImageSeg_CUDA/kernel.cu as NUM_OF_RANDOM_RESTARTS, feel free to alter it if you'd like.