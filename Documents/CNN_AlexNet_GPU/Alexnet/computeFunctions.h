#pragma once
#include <CL/cl.h>
#include <map>
#include <iostream>

#define DEVICES std::map<char*, DEVICE>

struct DEVICE
{
	int device_num;
	int platform;
	int device;
};

class ParallelCompute
{
public:
	cl_platform_id* pPlatformIDs;
	cl_device_id* pDeviceIDs;
	cl_context pContext;
	cl_command_queue pCommandQueue;
	cl_program pProgram;
	cl_kernel pKernel;
	cl_uint uiPlatformIDSize, uiDeviceIDSize;
public:
	ParallelCompute(DEVICE& dev);
	void destroy();
};

DEVICES getDevice();
char* showDevice(DEVICES& device_map);
void gpgpu_matrixMultiplication(ParallelCompute& cgu, float *A, float *B, float *Y, int A_row, int A_col, int B_row, int B_col);
void gpgpu_matrixMultiplication(ParallelCompute& cgu, float *A, float *B, float *B0, float *Y,  int A_row, int A_col, int B_row, int B_col);
void gpgpu_convolution(ParallelCompute& cgu, float *X, float *F, float *Y, int X_height, int X_width, int F_height, int F_width, int Y_height, int Y_width, int stride);
void gpgpu_convolution(ParallelCompute& cgu, float *X, float *F, float *Y, int X_depth, int X_height, int X_width, int F_height, int F_width, int Y_depth, int Y_height, int Y_width, int stride, int num);
void gpgpu_convolution(ParallelCompute& cgu, float *X, float *F, float *B0, float *Y, int X_depth, int X_height, int X_width, int F_height, int F_width, int Y_depth, int Y_height, int Y_width, int stride, int num);
