#include<iostream>
#include<CL/cl.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include"tga.h"

static const int MAX_NUM_PLATFORMS = 64;
static const int MAX_LENGTH_PLATFORM_NAME = 128;

// create a pixel struct for easy usage
typedef struct tag_Pixel
{
	unsigned char R;
	unsigned char G;
	unsigned char B;
} Pixel;

int main(void)
{
	// read the image from a file
	tga::TGAImage image;
	tga::LoadTGA(&image, "lenna.tga");

	// read the kernel from a file

	FILE *fp;
	char *source_str;
	size_t program_size;

	fopen_s(&fp, "RotationKernel.cl", "rb");
	if (!fp) {
		printf("Failed to load kernel\n");
		return 1;
	}

	fseek(fp, 0, SEEK_END);
	program_size = ftell(fp);
	rewind(fp);
	source_str = (char*)malloc(program_size + 1);
	source_str[program_size] = '\0';
	fread(source_str, sizeof(char), program_size, fp);
	fclose(fp);

	std::cout << source_str;

	cl_int ciErrNum;						// error number
	cl_device_id device;					// device id

	// get platforms
	cl_platform_id platforms[MAX_NUM_PLATFORMS];			// array for platforms
	cl_uint numPlatforms;									// number of platforms
	char platformName[MAX_LENGTH_PLATFORM_NAME];			// destination array for platform name
	size_t sizeRet;											// actual length of platform name
	ciErrNum = clGetPlatformIDs(64, platforms, &numPlatforms); // get them

	/*
	// print a list of all platforms
	printf("Platforms:\n");
	for (unsigned int i = 0; i < numPlatforms; ++i)
	{
		ciErrNum = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(char) * 120, platformName, &sizeRet);
		printf(platformName);
		printf("\n");
	}
	printf("\n");
	*/

	// get devices on platform 1 (in my case nvidia cuda, platform 0 is intel opencl)
	ciErrNum = clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_GPU, 1, &device, NULL);

	//create a context
	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

	//create a command queue
	cl_command_queue myqueue = clCreateCommandQueue(context, device, (cl_command_queue_properties)0, &ciErrNum);

	// create and build the program
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, NULL, &ciErrNum);
	ciErrNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	// create the kernel
	cl_kernel kernel = clCreateKernel(program, "image_rotate", &ciErrNum);

	/*
	// get information about the kernel to check if the right kernel function is called
	char kernelFunctionName[120];
	ciErrNum = clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 120, kernelFunctionName, &sizeRet);
	*/

	// create input and output arrays for the kernel function from the tga image
	unsigned int nNumChannels = 3; // TODO: if rgba this has to be 4 
	unsigned int nWidth = image.width;
	unsigned int nHeight = image.height;
	unsigned int nArraySize = nWidth * nHeight;

	Pixel* imageDataInput = new Pixel[nArraySize];
	Pixel* imageDataOutput = new Pixel[nArraySize];

	// transfer the tga image data to the pixel struct
	int j = 0;
	for (unsigned int i = 0; i < nArraySize * 3; i += 3)
	{
		imageDataInput[j].R = image.imageData[i];
		imageDataInput[j].G = image.imageData[i + 1];
		imageDataInput[j].B = image.imageData[i + 2];
		j++;
	}

	//create buffers for the image
	cl_mem d_ip = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(Pixel) * nArraySize, NULL, &ciErrNum);

	cl_mem d_op = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(Pixel) *  nArraySize, NULL, &ciErrNum);

	// write into the input buffer
	ciErrNum = clEnqueueWriteBuffer(myqueue, d_ip, CL_TRUE, 0, sizeof(Pixel) *  nArraySize, imageDataInput, 0, NULL, NULL);

	unsigned int nRotationPointX = nWidth / 2;
	unsigned int nRotationPointY = nHeight / 2;
	int degree = 180;
	float theta = degree * M_PI / 180;
	float sinTheta = sin(theta);
	float cosTheta = cos(theta);

	// set the arguments
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_ip);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_op);
	clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&nWidth);
	clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&nHeight);
	clSetKernelArg(kernel, 4, sizeof(cl_float), (void *)&sinTheta);
	clSetKernelArg(kernel, 5, sizeof(cl_float), (void *)&cosTheta);

	// set the workgroup sizes
	size_t globalws[] = { nWidth, nHeight};
	// execute kernel
	clEnqueueNDRangeKernel(myqueue, kernel, 2, 0, globalws, NULL, 0, NULL, NULL);

	// copy results from device back to host
	clEnqueueReadBuffer(myqueue, d_op, CL_TRUE, 0, sizeof(Pixel) * nArraySize, imageDataOutput, NULL, NULL, NULL);

	clFinish(myqueue);

	// create a vector from the output array
	std::vector<unsigned char> outputImage;

	j = 0;
	for (int i = 0; i < nArraySize * 3; i += nNumChannels)
	{
		outputImage.push_back(imageDataOutput[j].R);
		outputImage.push_back(imageDataOutput[j].G);
		outputImage.push_back(imageDataOutput[j].B);
		j++;
	}

	image.imageData = outputImage;
	tga::saveTGA(image, "lenna2.tga");

	return 0;
}