#include "computeFunctions.h"

void matMulTest();
void convolutionTest();
void fftTest();

int main() {
	matMulTest();
	//convolutionTest();
	//fftTest();

	getchar();

	return 0;
}

void matMulTest()
{
	//init
	int A_row, A_col, B_row, B_col;
	A_row = 1000;
	A_col = B_row = 1000;
	B_col = 1000;
	float *A = new float[A_row*A_col];
	for (int i = 0; i < A_row*A_col; i++) {
		A[i] = 1;
	}
	float *B = new float[B_row*B_col];
	for (int i = 0; i < B_row*B_col; i++) {
		B[i] = 1;
	}
	float *Y = new float[A_row*B_col];
	for (int i = 0; i < A_row*B_col; i++) {
		Y[i] = 0;
	}
	//test
	GPGPU tmp(CPU);
	auto begin = std::chrono::steady_clock::now();	
	for (int ep = 0; ep < 10; ep++) {
		float sum = 0;
		matrixMultiplication(tmp, A, B, Y, A_row, A_col, B_row, B_col);
		for (int i = 0; i < A_row*B_col; i++) {
			sum += Y[i];
		}
		std::cout << "Test Result: " << sum << std::endl;
	}
	auto end = std::chrono::steady_clock::now();
	tmp.destroy();
	std::cout << "Elapsed time for Test: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
}

void convolutionTest()
{
	//init
	int X_depth = 30, X_height = 50, X_width = 50;
	int Y_depth = 300, Y_height, Y_width;
	int F_height = 3, F_width = 3, stride = 1, num = 1;
	Y_height = (X_height - F_height)/stride + 1;
	Y_width = (X_width - F_width)/stride + 1;
	float *X = new float[X_depth * X_height * X_width * num];
	for (int i = 0; i < X_depth * X_height * X_width* num; i++) {
		X[i] = 1;
	}
	float *F = new float[X_depth * Y_depth * F_height * F_width * num];
	for (int i = 0; i < X_depth * Y_depth * F_height * F_width* num; i++) {
		F[i] = 1;
	}
	float *Y = new float[Y_depth * Y_height * Y_width * num];
	for (int i = 0; i < Y_depth * Y_height * Y_width* num; i++) {
		Y[i] = 0;
	}
	//test
	GPGPU tmp(GPU);
	auto begin = std::chrono::steady_clock::now();
	float sum = 0;
	convolution(tmp, X, F, Y, X_depth, X_height, X_width, F_height, F_width, Y_depth, Y_height, Y_width, stride, num);
	for (int i = 0; i < Y_depth * Y_height * Y_width* num; i++) {
		sum += Y[i];
	}
	std::cout << "Test Result: " << sum << std::endl;
	auto end = std::chrono::steady_clock::now();
	tmp.destroy();
	std::cout << "Elapsed time for Test: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
}

void fftTest()
{

}


