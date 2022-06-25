#pragma once
#include <CL/cl.h>
#include <chrono>
#include <iostream>

#define CPU 2909
#define GPU 8813

class GPGPU
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
	GPGPU(int flag);
	void destroy();
};

void matrixMultiplication(GPGPU cgu, float *A, float *B, float *Y, int A_row, int A_col, int B_row, int B_col);
void convolution(GPGPU cgu, float *X, float *F, float *Y, int X_height, int X_width, int F_height, int F_width, int Y_height, int Y_width, int stride);
void convolution(GPGPU cgu, float *X, float *F, float *Y, int X_depth, int X_height, int X_width, int F_height, int F_width, int Y_depth, int Y_height, int Y_width, int stride, int num);
