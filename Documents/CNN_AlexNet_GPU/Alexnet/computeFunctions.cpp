#include "computeFunctions.h"

char *get_source_code(const char *file_name, size_t *len) {
	char *source_code;
	size_t length;
	FILE *file = fopen(file_name, "r");
	if (file == NULL) {
		printf("[%s:%d] OpenCL error %d/\n", __FILE__, __LINE__, file_name);
		exit(EXIT_FAILURE);
	}

	fseek(file, 0, SEEK_END);
	length = (size_t)ftell(file);
	rewind(file);

	source_code = (char *)malloc(length + 1);
	fread(source_code, length, 1, file);
	source_code[length] = '\0';

	fclose(file);
	*len = length;
	return source_code;
}

DEVICES getDevice()
{
	DEVICES device_map;
	size_t valueSize;
	cl_uint platformCount;
	cl_platform_id* platforms;
	cl_uint deviceCount;
	cl_device_id* devices;
	cl_uint maxComputeUnits;
	int device_num = 0;
	//플랫폼 반환
	clGetPlatformIDs(0, NULL, &platformCount);
	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platformCount);
	clGetPlatformIDs(platformCount, platforms, NULL);
	for (int i = 0; i < platformCount; i++) {
		//디바이스 반환
		clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
		devices = (cl_device_id*)malloc(sizeof(cl_device_id) * deviceCount);
		clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);
		for (int j = 0; j < deviceCount; j++) {
			DEVICE dev;
			dev.device_num = device_num++;
			dev.platform = i;
			dev.device = j;
			//디바이스 이름 반환
			clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
			char* value = (char*)malloc(valueSize);
			clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
			device_map[value] = dev;
		}
	}
	return device_map;
}

char* showDevice(DEVICES& device_map)
{
	int num;
	std::string device_name_str;
	for (auto iter = device_map.begin(); iter != device_map.end(); iter++)
	{
		std::cout << "[" << iter->second.device_num << "] " << iter->first << std::endl;
	}
	std::cin >> num;
	for (auto iter = device_map.begin(); iter != device_map.end(); iter++)
	{
		if (iter->second.device_num == num) return iter->first;
	}
	return "";
}

ParallelCompute::ParallelCompute(DEVICE& dev)
{
	cl_int nError;
	cl_int nProgramError = -1;
	while (nProgramError != CL_SUCCESS) {
		//사용 가능한 플랫폼 개수 반환
		if (clGetPlatformIDs(0, NULL, &uiPlatformIDSize) != CL_SUCCESS)
		{
			std::cout << "Error - Line " << __LINE__ - 2 << std::endl;
		}
		//플랫폼 갯수만큼 id를 new 생성
		pPlatformIDs = new cl_platform_id[uiPlatformIDSize];
		if (pPlatformIDs == 0)
		{
			std::cout << "Error - Line " << __LINE__ - 3 << std::endl;
		}
		//플랫폼 id 목록 반환
		if (clGetPlatformIDs(uiPlatformIDSize, pPlatformIDs, NULL) != CL_SUCCESS)
		{
			std::cout << "Error - Line " << __LINE__ - 2 << std::endl;
		}
		//디바이스 생성
		if (clGetDeviceIDs(pPlatformIDs[dev.platform], CL_DEVICE_TYPE_ALL, 0, NULL, &uiDeviceIDSize) != CL_SUCCESS)
		{
			std::cout << "Error - Line " << __LINE__ - 2 << std::endl;
		}
		//디바이스 갯수만큼 디바이스 메모리 할당
		pDeviceIDs = new cl_device_id[uiDeviceIDSize];
		if (pDeviceIDs == 0)
		{
			std::cout << "Error - Line " << __LINE__ - 3 << std::endl;
		}
		//디바이스 반환
		if (clGetDeviceIDs(pPlatformIDs[dev.platform], CL_DEVICE_TYPE_ALL, uiDeviceIDSize, pDeviceIDs, NULL) != CL_SUCCESS)
		{
			std::cout << "Error - Line " << __LINE__ - 2 << std::endl;
		}
		//콘텍스트 생성
		cl_context_properties Properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)pPlatformIDs[dev.platform], 0, };
		pContext = clCreateContext(Properties, uiDeviceIDSize, pDeviceIDs, NULL, NULL, &nError);
		if (pContext == 0 || nError != CL_SUCCESS)
		{
			std::cout << "Error - Line " << __LINE__ - 3 << std::endl;
		}
		//커맨트 큐 생성, 첫번째 디바이스 선택
		pCommandQueue = clCreateCommandQueueWithProperties(pContext, pDeviceIDs[dev.device], 0, &nError);
		if (pCommandQueue == 0 || nError != CL_SUCCESS)
		{
			std::cout << "Error - Line " << __LINE__ - 3 << std::endl;

		}
		//프로그램 생성
		char *uiSourceCode;
		size_t uiSourceLength;
		uiSourceCode = get_source_code("computeFunctions.cl", &uiSourceLength);
		pProgram = clCreateProgramWithSource(pContext, 1, (const char **)&uiSourceCode, &uiSourceLength, &nError);
		if (pProgram == 0 || nError != CL_SUCCESS)
		{
			std::cout << "Error - Line " << __LINE__ - 3 << std::endl;
		}
		//프로그램 빌드
		nProgramError = clBuildProgram(pProgram, uiDeviceIDSize, pDeviceIDs, 0, 0, 0);
		if (nProgramError != CL_SUCCESS)
		{
			std::cout << "Error - Line " << __LINE__ - 3 << std::endl;
		}
	}
}

void ParallelCompute::destroy()
{
	//플랫폼 해제
	if (pPlatformIDs != 0) delete[] pPlatformIDs;
	//디바이스 해제
	if (pDeviceIDs != 0) delete[] pDeviceIDs;
	//커널 해제
	if (pKernel) clReleaseKernel(pKernel);
	//커맨드 큐 해제
	if (pCommandQueue) clReleaseCommandQueue(pCommandQueue);
	//컨택스트 해제
	if (pContext) clReleaseContext(pContext);
	//프로그램 해제
	if (pProgram) clReleaseProgram(pProgram);
}

void gpgpu_matrixMultiplication(ParallelCompute& cgu, float *A, float *B, float *Y, int A_row, int A_col, int B_row, int B_col)
{
	if (A_row*B_col < 10000)
	{
		for (int i = 0; i < A_row; i++) {
			for (int j = 0; j < B_col; j++) {
				for (int k = 0; k < A_col; k++) {
					Y[i*B_col + j] += A[i*A_col + k] * B[k*B_col + j];
				}
			}
		}
	}
	else
	{
		cl_int nError;
		//커널 생성
		cgu.pKernel = clCreateKernel(cgu.pProgram, "matMul", &nError);
		if (cgu.pKernel == 0 || nError != CL_SUCCESS)
		{
			std::cout << "Error - Line " << __LINE__ - 3 << std::endl;
		}
		//Matrix 사이즈 저장
		int *size = (int*)malloc(sizeof(int) * 4);
		size[0] = A_row; size[1] = A_col; size[2] = B_row; size[3] = B_col;
		//커널 실행용 버퍼 생성
		cl_mem pInputBufA = 0, pInputBufB = 0, pOutputBufY = 0, pSizeBuf = 0;
		pInputBufA = clCreateBuffer(cgu.pContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int) * A_row * A_col, A, &nError);
		pInputBufB = clCreateBuffer(cgu.pContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int) * B_row * B_col, B, &nError);
		pOutputBufY = clCreateBuffer(cgu.pContext, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int) * A_row * B_col, Y, &nError);
		pSizeBuf = clCreateBuffer(cgu.pContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int) * 4, size, &nError);
		//work size 설정
		size_t global_size[2] = { B_col,A_row };
		size_t local_size[2] = { 16,16 };
		global_size[0] = (global_size[0] + local_size[0] - 1) / local_size[0] * local_size[0];
		global_size[1] = (global_size[1] + local_size[1] - 1) / local_size[1] * local_size[1];
		//커널에 들어갈 인자 설정
		clSetKernelArg(cgu.pKernel, 0, sizeof(cl_mem), (void*)&pInputBufA);
		clSetKernelArg(cgu.pKernel, 1, sizeof(cl_mem), (void*)&pInputBufB);
		clSetKernelArg(cgu.pKernel, 2, sizeof(cl_mem), (void *)&pOutputBufY);
		clSetKernelArg(cgu.pKernel, 3, sizeof(cl_mem), (void*)&pSizeBuf);
		//커널 실행
		nError = clEnqueueNDRangeKernel(cgu.pCommandQueue, cgu.pKernel, 2, 0, global_size, local_size, 0, NULL, NULL);
		if (nError != CL_SUCCESS)
		{
			std::cout << "Error - Line " << __LINE__ - 3 << std::endl;
		}
		//커널 실행 종료 후 버퍼 읽기
		nError = clEnqueueReadBuffer(cgu.pCommandQueue, pOutputBufY, CL_TRUE, 0, sizeof(int) * A_row * B_col, Y, 0, NULL, NULL);
		if (nError != CL_SUCCESS)
		{
			std::cout << "Error - Line " << __LINE__ - 3 << std::endl;
		}
		//버퍼 해제
		if (pInputBufA) clReleaseMemObject(pInputBufA);
		if (pInputBufB) clReleaseMemObject(pInputBufB);
		if (pOutputBufY) clReleaseMemObject(pOutputBufY);
		if (pSizeBuf) clReleaseMemObject(pSizeBuf);
		delete[] size;
	}
}

void gpgpu_matrixMultiplication(ParallelCompute& cgu, float *A, float *B, float *B0, float *Y, int A_row, int A_col, int B_row, int B_col)
{
	if (A_row*B_col < 10000)
	{
		for (int i = 0; i < A_row; i++) {
			for (int j = 0; j < B_col; j++) {
				for (int k = 0; k < A_col; k++) {
					Y[i*B_col + j] += A[i*A_col + k] * B[k*B_col + j];
				}
				Y[i*B_col + j] += B0[i*B_col + j];
			}
		}
	}
	else
	{
		cl_int nError;
		//커널 생성
		cgu.pKernel = clCreateKernel(cgu.pProgram, "matMulwithBias", &nError);
		if (cgu.pKernel == 0 || nError != CL_SUCCESS)
		{
			std::cout << "Error - Line " << __LINE__ - 3 << std::endl;
		}
		//Matrix 사이즈 저장
		int *size = (int*)malloc(sizeof(int) * 4);
		size[0] = A_row; size[1] = A_col; size[2] = B_row; size[3] = B_col;
		//커널 실행용 버퍼 생성
		cl_mem pInputBufA = 0, pInputBufB = 0, pInputBufB0 = 0, pOutputBufY = 0, pSizeBuf = 0;
		pInputBufA = clCreateBuffer(cgu.pContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int) * A_row * A_col, A, &nError);
		pInputBufB = clCreateBuffer(cgu.pContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int) * B_row * B_col, B, &nError);
		pInputBufB0 = clCreateBuffer(cgu.pContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int) * A_row * B_col, B0, &nError);
		pOutputBufY = clCreateBuffer(cgu.pContext, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int) * A_row * B_col, Y, &nError);
		pSizeBuf = clCreateBuffer(cgu.pContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int) * 4, size, &nError);
		//work size 설정
		size_t global_size[2] = { B_col,A_row };
		size_t local_size[2] = { 16,16 };
		global_size[0] = (global_size[0] + local_size[0] - 1) / local_size[0] * local_size[0];
		global_size[1] = (global_size[1] + local_size[1] - 1) / local_size[1] * local_size[1];
		//커널에 들어갈 인자 설정
		clSetKernelArg(cgu.pKernel, 0, sizeof(cl_mem), (void*)&pInputBufA);
		clSetKernelArg(cgu.pKernel, 1, sizeof(cl_mem), (void*)&pInputBufB);
		clSetKernelArg(cgu.pKernel, 2, sizeof(cl_mem), (void*)&pInputBufB0);
		clSetKernelArg(cgu.pKernel, 3, sizeof(cl_mem), (void *)&pOutputBufY);
		clSetKernelArg(cgu.pKernel, 4, sizeof(cl_mem), (void*)&pSizeBuf);
		//커널 실행
		nError = clEnqueueNDRangeKernel(cgu.pCommandQueue, cgu.pKernel, 2, 0, global_size, local_size, 0, NULL, NULL);
		if (nError != CL_SUCCESS)
		{
			std::cout << "Error - Line " << __LINE__ - 3 << std::endl;
		}
		//커널 실행 종료 후 버퍼 읽기
		nError = clEnqueueReadBuffer(cgu.pCommandQueue, pOutputBufY, CL_TRUE, 0, sizeof(int) * A_row * B_col, Y, 0, NULL, NULL);
		if (nError != CL_SUCCESS)
		{
			std::cout << "Error - Line " << __LINE__ - 3 << std::endl;
		}
		//버퍼 해제
		if (pInputBufA) clReleaseMemObject(pInputBufA);
		if (pInputBufB) clReleaseMemObject(pInputBufB);
		if (pInputBufB0) clReleaseMemObject(pInputBufB0);
		if (pOutputBufY) clReleaseMemObject(pOutputBufY);
		if (pSizeBuf) clReleaseMemObject(pSizeBuf);
		delete[] size;
	}
}

void gpgpu_convolution(ParallelCompute& cgu, float *X, float *F, float *Y, int X_height, int X_width, int F_height, int F_width, int Y_height, int Y_width, int stride)
{
	if (Y_height*Y_width < 10000)
	{
		for (int i = 0; i < Y_height; i++) {
			for (int j = 0; j < Y_width; j++) {
				for (int m = 0; m < F_height; m++) {
					for (int n = 0; n < F_width; n++) {
						Y[i*Y_width + j] += X[(stride*i + m)*X_width + (stride*j + n)] * F[m*F_width + n];
					}
				}
			}
		}
	}
	else
	{
		cl_int nError;
		//커널 생성
		cgu.pKernel = clCreateKernel(cgu.pProgram, "convolution2D", &nError);
		if (cgu.pKernel == 0 || nError != CL_SUCCESS)
		{
			std::cout << "Error - Line " << __LINE__ - 3 << std::endl;
		}
		//Matrix 사이즈 저장
		int *size = (int*)malloc(sizeof(int) * 7);
		size[0] = X_height; size[1] = X_width; size[2] = Y_height; size[3] = Y_width, size[4] = F_height; size[5] = F_width; size[6] = stride; 
		//커널 실행용 버퍼 생성
		cl_mem pInputBufX = 0, pInputBufF = 0, pOutputBufY = 0, pSizeBuf = 0;
		pInputBufX = clCreateBuffer(cgu.pContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int) *  X_height * X_width, X, &nError);
		pInputBufF = clCreateBuffer(cgu.pContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int) *  F_height * F_width, F, &nError);
		pOutputBufY = clCreateBuffer(cgu.pContext, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int) * Y_height * Y_width, Y, &nError);
		pSizeBuf = clCreateBuffer(cgu.pContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int) * 7, size, &nError);
		//work size 설정
		size_t global_size[2] = { Y_width , Y_height };
		size_t local_size[2] = { 16, 16 };
		global_size[0] = (global_size[0] + local_size[0] - 1) / local_size[0] * local_size[0];
		global_size[1] = (global_size[1] + local_size[1] - 1) / local_size[1] * local_size[1];
		//커널에 들어갈 인자 설정
		clSetKernelArg(cgu.pKernel, 0, sizeof(cl_mem), (void*)&pInputBufX);
		clSetKernelArg(cgu.pKernel, 1, sizeof(cl_mem), (void*)&pInputBufF);
		clSetKernelArg(cgu.pKernel, 2, sizeof(cl_mem), (void *)&pOutputBufY);
		clSetKernelArg(cgu.pKernel, 3, sizeof(cl_mem), (void*)&pSizeBuf);
		//커널 실행
		clEnqueueNDRangeKernel(cgu.pCommandQueue, cgu.pKernel, 2, 0, global_size, local_size, 0, NULL, NULL);
		if (nError != CL_SUCCESS)
		{
			std::cout << "Error - Line " << __LINE__ - 3 << std::endl;
		}
		//커널 실행 종료 후 버퍼 읽기
		clEnqueueReadBuffer(cgu.pCommandQueue, pOutputBufY, CL_TRUE, 0, sizeof(int) * Y_height * Y_width, Y, 0, NULL, NULL);
		if (nError != CL_SUCCESS)
		{
			std::cout << "Error - Line " << __LINE__ - 3 << std::endl;
		}
		//버퍼 해제
		if (pInputBufX) clReleaseMemObject(pInputBufX);
		if (pInputBufF) clReleaseMemObject(pInputBufF);
		if (pOutputBufY) clReleaseMemObject(pOutputBufY);
		if (pSizeBuf) clReleaseMemObject(pSizeBuf);
		delete[] size;
	}
}

void gpgpu_convolution(ParallelCompute& cgu, float *X, float *F, float *Y, int X_depth, int X_height, int X_width, int F_height, int F_width, int Y_depth, int Y_height, int Y_width, int stride, int num)
{
	if (Y_depth*Y_height*Y_width*num < 10000)
	{
		for (int i = 0; i < num * Y_depth; i++) {
			for (int j = 0; j < Y_height * Y_width; j++) {
				for (int k = (i / Y_depth) * X_depth; k < (i / Y_depth + 1) * X_depth; k++) {
					for (int m = 0; m < F_height; m++) {
						for (int n = 0; n < F_width; n++) {
							Y[i*Y_height*Y_width + j] += X[k*X_height*X_width + (stride*(j / Y_width) + m)*X_width + (stride*(j%Y_width) + n)] * F[k*Y_depth*F_height*F_width + (i%Y_depth)*F_height*F_width + m*F_width + n];
						}
					}
				}
			}
		}
	}
	else
	{
		cl_int nError;
		//커널 생성
		cgu.pKernel = clCreateKernel(cgu.pProgram, "convolution3D", &nError);
		if (cgu.pKernel == 0 || nError != CL_SUCCESS)
		{
			std::cout << "Error - Line " << __LINE__ - 3 << std::endl;
		}
		//Matrix 사이즈 저장
		int *size = (int*)malloc(sizeof(int) * 10);
		size[0] = X_depth; size[1] = X_height; size[2] = X_width; size[3] = Y_depth; size[4] = Y_height; size[5] = Y_width, size[6] = F_height; size[7] = F_width; size[8] = stride; size[9] = num;
		//커널 실행용 버퍼 생성
		cl_mem pInputBufX = 0, pInputBufF = 0, pOutputBufY = 0, pSizeBuf = 0;
		pInputBufX = clCreateBuffer(cgu.pContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int) * num * X_depth * X_height * X_width, X, &nError);
		pInputBufF = clCreateBuffer(cgu.pContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int) * num * X_depth * Y_depth * F_height * F_width, F, &nError);
		pOutputBufY = clCreateBuffer(cgu.pContext, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int) * num * Y_depth * Y_height * Y_width, Y, &nError);
		pSizeBuf = clCreateBuffer(cgu.pContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int) * 10, size, &nError);
		//work size 설정
		size_t global_size[2] = { Y_height * Y_width ,  Y_depth * num };
		size_t local_size[2] = { 16, 16 };
		global_size[0] = (global_size[0] + local_size[0] - 1) / local_size[0] * local_size[0];
		global_size[1] = (global_size[1] + local_size[1] - 1) / local_size[1] * local_size[1];
		//커널에 들어갈 인자 설정
		clSetKernelArg(cgu.pKernel, 0, sizeof(cl_mem), (void*)&pInputBufX);
		clSetKernelArg(cgu.pKernel, 1, sizeof(cl_mem), (void*)&pInputBufF);
		clSetKernelArg(cgu.pKernel, 2, sizeof(cl_mem), (void *)&pOutputBufY);
		clSetKernelArg(cgu.pKernel, 3, sizeof(cl_mem), (void*)&pSizeBuf);
		//커널 실행
		clEnqueueNDRangeKernel(cgu.pCommandQueue, cgu.pKernel, 2, 0, global_size, local_size, 0, NULL, NULL);
		if (nError != CL_SUCCESS)
		{
			std::cout << "Error - Line " << __LINE__ - 3 << std::endl;
		}
		//커널 실행 종료 후 버퍼 읽기
		clEnqueueReadBuffer(cgu.pCommandQueue, pOutputBufY, CL_TRUE, 0, sizeof(int) * num * Y_depth * Y_height * Y_width, Y, 0, NULL, NULL);
		if (nError != CL_SUCCESS)
		{
			std::cout << "Error - Line " << __LINE__ - 3 << std::endl;
		}
		//버퍼 해제
		if (pInputBufX) clReleaseMemObject(pInputBufX);
		if (pInputBufF) clReleaseMemObject(pInputBufF);
		if (pOutputBufY) clReleaseMemObject(pOutputBufY);
		if (pSizeBuf) clReleaseMemObject(pSizeBuf);
		delete[] size;
	}	
}

void gpgpu_convolution(ParallelCompute& cgu, float *X, float *F, float *B0, float *Y, int X_depth, int X_height, int X_width, int F_height, int F_width, int Y_depth, int Y_height, int Y_width, int stride, int num)
{
	if (Y_depth*Y_height*Y_width*num < 10000)
	{
		for (int i = 0; i < num * Y_depth; i++) {
			for (int j = 0; j < Y_height * Y_width; j++) {
				for (int k = (i / Y_depth) * X_depth; k < (i / Y_depth + 1) * X_depth; k++) {
					for (int m = 0; m < F_height; m++) {
						for (int n = 0; n < F_width; n++) {
							Y[i*Y_height*Y_width + j] += X[k*X_height*X_width + (stride*(j / Y_width) + m)*X_width + (stride*(j%Y_width) + n)] * F[k*Y_depth*F_height*F_width + (i%Y_depth)*F_height*F_width + m*F_width + n];
						}
					}
				}
				Y[i*Y_height*Y_width + j] += B0[i*Y_height*Y_width + j];
			}
		}
	}
	else
	{
		cl_int nError;
		//커널 생성
		cgu.pKernel = clCreateKernel(cgu.pProgram, "convolution3DwithBias", &nError);
		if (cgu.pKernel == 0 || nError != CL_SUCCESS)
		{
			std::cout << "Error - Line " << __LINE__ - 3 << std::endl;
		}
		//Matrix 사이즈 저장
		int *size = (int*)malloc(sizeof(int) * 10);
		size[0] = X_depth; size[1] = X_height; size[2] = X_width; size[3] = Y_depth; size[4] = Y_height; size[5] = Y_width, size[6] = F_height; size[7] = F_width; size[8] = stride; size[9] = num;
		//커널 실행용 버퍼 생성
		cl_mem pInputBufX = 0, pInputBufF = 0, pInputBufB0 = 0, pOutputBufY = 0, pSizeBuf = 0;
		pInputBufX = clCreateBuffer(cgu.pContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int) * num * X_depth * X_height * X_width, X, &nError);
		pInputBufF = clCreateBuffer(cgu.pContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int) * num * X_depth * Y_depth * F_height * F_width, F, &nError);
		pInputBufB0 = clCreateBuffer(cgu.pContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int) * num * Y_depth * Y_height * Y_width, B0, &nError);
		pOutputBufY = clCreateBuffer(cgu.pContext, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int) * num * Y_depth * Y_height * Y_width, Y, &nError);
		pSizeBuf = clCreateBuffer(cgu.pContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int) * 10, size, &nError);
		//work size 설정
		size_t global_size[2] = { Y_height * Y_width ,  Y_depth * num };
		size_t local_size[2] = { 16, 16 };
		global_size[0] = (global_size[0] + local_size[0] - 1) / local_size[0] * local_size[0];
		global_size[1] = (global_size[1] + local_size[1] - 1) / local_size[1] * local_size[1];
		//커널에 들어갈 인자 설정
		clSetKernelArg(cgu.pKernel, 0, sizeof(cl_mem), (void*)&pInputBufX);
		clSetKernelArg(cgu.pKernel, 1, sizeof(cl_mem), (void*)&pInputBufF);
		clSetKernelArg(cgu.pKernel, 2, sizeof(cl_mem), (void *)&pInputBufB0);
		clSetKernelArg(cgu.pKernel, 3, sizeof(cl_mem), (void *)&pOutputBufY);
		clSetKernelArg(cgu.pKernel, 4, sizeof(cl_mem), (void*)&pSizeBuf);
		//커널 실행
		clEnqueueNDRangeKernel(cgu.pCommandQueue, cgu.pKernel, 2, 0, global_size, local_size, 0, NULL, NULL);
		if (nError != CL_SUCCESS)
		{
			std::cout << "Error - Line " << __LINE__ - 3 << std::endl;
		}
		//커널 실행 종료 후 버퍼 읽기
		clEnqueueReadBuffer(cgu.pCommandQueue, pOutputBufY, CL_TRUE, 0, sizeof(int) * num * Y_depth * Y_height * Y_width, Y, 0, NULL, NULL);
		if (nError != CL_SUCCESS)
		{
			std::cout << "Error - Line " << __LINE__ - 3 << std::endl;
		}
		//버퍼 해제
		if (pInputBufX) clReleaseMemObject(pInputBufX);
		if (pInputBufF) clReleaseMemObject(pInputBufF);
		if (pInputBufB0) clReleaseMemObject(pInputBufB0);
		if (pOutputBufY) clReleaseMemObject(pOutputBufY);
		if (pSizeBuf) clReleaseMemObject(pSizeBuf);
		delete[] size;
	}
}