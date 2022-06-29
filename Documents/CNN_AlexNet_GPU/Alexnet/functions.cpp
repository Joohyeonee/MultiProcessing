#include "functions.h"
#include <random>
#include <fstream>


vector<vector<vector<double>>> setImage(string& path)
{
	Mat img, re_img;
	img = cv::imread(path, cv::IMREAD_COLOR);

	vector<vector<vector<double>>> tmp;
	resizeVec(tmp, 3, 227, 227);
	resize(img, re_img, Size(227, 227));

	int Nrow = 227;
	int Ncol = 227;
	int Ndep = 3;

	for (int i = 0; i < Ndep; i++) {
		for (int j = 0; j < Nrow; j++) {
			for (int k = 0; k < Ncol; k++) {
				tmp[i][j][k] = re_img.at<cv::Vec3b>(j, k)[i];
			}
		}
	}
	
	NormalizeImage(tmp);
	return tmp;
}

vector<String> getImage(string& path) {
	vector<String> filenames;
	glob(path, filenames, true);
	return filenames;
}

vector<vector<double>> setTarget(string& imagePath) {
	vector<vector<double>> tmp;
	resizeVec(tmp, 1, 1000);
	int idx = imagePath.find('(') + 1;
	string str_num = imagePath.substr(idx, 1);
	int num = stoi(str_num);
	tmp[0][num - 1] = 1;
	return tmp;
}

vector<vector<vector<double>>> convolution(ParallelCompute pc, vector<vector<vector<double>>> &input, vector<vector<vector<double>>> &bias, vector<vector<vector<vector<double>>>> &filter,int stride) {
	int input_depth = input.size();
	int input_height = input[0].size();
	int input_width = input[0][0].size();
	int output_depth = filter.size();
	int filter_height = filter[0][0].size(); 
	int filter_width = filter[0][0][0].size(); 
	int outSize = ((input_height - filter_height) / stride ) + 1;
	vector<vector<vector<double>>> output;
	resizeVec(output, output_depth, outSize, outSize);

	float *pOutput = new float[output_depth * outSize * outSize]();
	float *pInput = changeTo1D(input);
	float *pFilter = changeTo1D(filter);
	gpgpu_convolution(pc, pInput, pFilter, pOutput, input_depth, input_height, input_width, filter_height, filter_width, output_depth, outSize, outSize, stride, 1);
	changeTo3D(output, pOutput);
	for (int i = 0; i < output_depth; i++) {
		for (int j = 0; j < outSize; j++) {
			for (int k = 0; k < outSize; k++) {
				output[i][j][k] += bias[i][0][0];
			}
		}
	}
	delete[] pOutput;
	delete[] pInput;
	delete[] pFilter;
	return output;
}

vector<vector<vector<double>>> maxPooling(vector<vector<vector<double>>> &activated, vector<vector<vector<double>>> &position1, vector<vector<vector<double>>> &position2, int poolsize ,int stride) {
	int X_depth = activated.size();
	int X_height = activated[0].size();
	int outSize = ((X_height - poolsize) / stride) + 1;
	vector<vector<vector<double>>> tmp;
	resizeVec(tmp,X_depth, outSize, outSize);
	int I = tmp.size();
	int J = tmp[0].size();
	int K = tmp[0][0].size();
	for (int i = 0; i <	I; i++) {
		for (int j = 0; j < J; j++) {
			for (int k = 0; k < K; k++) {
				double max = 0;
				int x = j*stride;
				int y = k*stride;
				for (int l = 0; l < poolsize; l++) {
					for (int m = 0; m < poolsize; m++) {
						if (max < activated[i][j * stride + l][k * stride + m]) {
							max = activated[i][j * stride + l][k * stride + m];
							tmp[i][j][k] = max;
							x = j*stride + l;
							y = k*stride + m;
						}
					}
				}
				position1[i][j][k] = x;
				position2[i][j][k] = y;
			}
		}
	}
	return tmp;
}

vector<vector<double>> flatten(vector<vector<vector<double>>> &input) {
	int I = input.size();
	int J = input[0].size();
	int K = input[0][0].size();
	vector<vector<double>> tmp;
	resizeVec(tmp,1, I*J*K);
	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {  
			for (int k = 0; k < K; k++) {
				tmp[0][k+((K*i + j)*K)] = input[i][j][k];
			}
		}
	}
	return tmp;
}

void multiplication(ParallelCompute pc, vector<vector<double>> &input, vector<vector<double>> &output, vector<vector<double>> &bias, vector<vector<double>> &weight) {
	int output_nodes = output[0].size();
	int input_nodes = input[0].size();

	float *pOutput = new float[output_nodes]();
	float *pInput = changeTo1D(input);
	float *pWeight = changeTo1D(weight);
	float *pBias = changeTo1D(bias);
	gpgpu_matrixMultiplication(pc, pInput, pWeight, pBias, pOutput, 1, input_nodes, input_nodes, output_nodes);
	changeTo2D(output, pOutput);
	delete[] pOutput;
	delete[] pInput;
	delete[] pWeight;
	delete[] pBias;
}

vector<vector<double>> loss(vector<vector<double>> &output, vector<vector<double>> &target) {
	vector<vector<double>> tmp;
	resizeVec(tmp,1, output[0].size());
	for (int i = 0; i < output[0].size(); i++) {
		tmp[0][i] = output[0][i] - target[0][i];
	}
	return tmp;
}

vector<vector<double>> calError(vector<vector<double>> &errorIn, vector<vector<double>> &input) {
	int I = errorIn.size();
	int	J = errorIn[0].size();
	vector<vector<double>> tmp;
	resizeVec(tmp,I, J);
	vector<vector<double>> tmp2 = dreluF(input);

	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			tmp[i][j] = errorIn[i][j] * tmp2[i][j];
		}
	}
	return tmp;
}

vector<vector<double>> calWeightDiff(vector<vector<double>> &bias, vector<vector<double>> &error, vector<vector<double>> &conv){
	int I = conv[0].size();
	int	J = error[0].size();
	int K = bias[0].size();
	vector<vector<double>> tmp;
	resizeVec(tmp,I, J);
	double learninglate = 0.0001;
	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			tmp[i][j] = learninglate * error[0][j] * conv[0][i];
		}
	}
	return tmp;
}

vector<vector<double>> calError_in(ParallelCompute pc, vector<vector<double>> &error, vector<vector<double>> &weight) {	
	int E_height = error.size();
	int I = weight.size();
	int J = weight[0].size();
	vector<vector<double>> tmp;
	resizeVec(tmp,E_height, I);
	for (int i = 0; i < I;i++) { 
		for (int j = 0; j < J;j++) { 
			tmp[0][i] += error[0][j] * weight[i][j]; 
		}
	}
	return tmp;
}

void updateBias(vector<vector<double>> &bias, vector<vector<double>> &error)
{
	double learninglate = 0.001;
	for (int i = 0; i < bias[0].size(); i++) {
		bias[0][i] -= learninglate * error[0][i];
	}
}

void updateWeight(vector<vector<double>> &weight, vector<vector<double>> &deltaWeight) {
	int I = weight.size();
	int	J = weight[0].size();
	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			weight[i][j] -= deltaWeight[i][j];
			deltaWeight[i][j] = 0;
		}
	}
}

vector<vector<vector<double>>> dflatten(vector<vector<double>> &errorIn , int width) {
	int X_width = errorIn[0].size();
	vector<vector<vector<double>>> tmp;
	resizeVec(tmp,X_width / (width * width), width, width);
	int I = tmp.size();
	int J = tmp[0].size();
	int K = tmp[0][0].size();
	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			for (int k = 0; k < K; k++) {
				tmp[i][j][k] = errorIn[0][k + ((K*i + j)*K)];
			}
		}
	}
	return tmp;
}

vector<vector<vector<double>>> calError(vector<vector<vector<double>>> &errorIn, vector<vector<vector<double>>> &conv) {
	int I = conv.size();
	int J = conv[0].size();
	int K = conv[0][0].size();
	vector<vector<vector<double>>> tmp;
	resizeVec(tmp,I, J, K);
	vector<vector<vector<double>>> tmp2;
	resizeVec(tmp2, I, J, K);
	tmp2 = dreluF(conv);
	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			for (int k = 0; k < K; k++) {
				tmp[i][j][k] = errorIn[i][j][k] * tmp2[i][j][k];
			}
		}
	}
	return tmp;
}

vector<vector<vector<double>>> poolError(vector<vector<vector<double>>> &error, vector<vector<vector<double>>> &position1, vector<vector<vector<double>>> &position2, int stride, int num) {
	int prev_height = ((position1.size()-1) * stride) + num;
	int prev_width = ((position1[0].size()-1) * stride) + num;
	int post_depth = error.size();
	int post_height = error[0].size();
	int post_width = error[0][0].size();

	vector<vector<vector<double>>> tmp;
	resizeVec(tmp, post_depth, prev_height, prev_width);
   for (int i = 0; i < post_depth; i++) {
      for (int j = 0; j < post_height; j++) {
         for (int k = 0; k < post_width; k++) {
			 double xPos = 0;
			 double yPos = 0;
			 for (int l = 0; l < num; l++) {
				 for (int m = 0; m < num; m++) {
					 xPos = position1[i][j][k];
					 yPos = position2[i][j][k];
					 tmp[i][xPos][yPos] = error[i][j][k];
				 }
			 }
         }
      }
   }
   return tmp;
}

void updateBias(vector<vector<vector<double>>> &bias, vector<vector<vector<double>>> &delta)
{
	double learningrate = 0.001;
	for (int i = 0; i < bias.size(); i++) {
		for (int j = 0; j < bias[0].size(); j++) {
			for (int k = 0; k < bias[0][0].size(); k++) {
				bias[i][j][k] -= delta[i][j][k];
			}
		}
	}
}


void updateBias2(vector<vector<double>> &bias, vector<vector<double>> &delta)
{
	double learningrate = 0.001;
	for (int i = 0; i < bias.size(); i++) {
		for (int j = 0; j < bias[0].size(); j++) {
			bias[i][j] -= delta[i][j];
		}
	}
}

vector<vector<vector<vector<double>>>> calFilterDiff(vector<vector<vector<vector<double>>>> &filter, vector<vector<vector<double>>> &bias, vector<vector<vector<double>>> &error, vector<vector<vector<double>>> &padded, int stride) {
	int I = filter.size();
	int J = filter[0].size();
	int K = filter[0][0].size();
	int L = filter[0][0][0].size();
	int M = error[0].size();
	int N = error[0][0].size();
	vector<vector<vector<vector<double>>>> tmp; 
	resizeVec(tmp,I, J, K, L);
	double learningrate = 0.001;	
	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			for (int k = 0; k < K; k++) {
				for (int l = 0; l < L; l++) {
					for (int m = 0; m < M; m++) {
						for (int n = 0; n < N; n++) {
							tmp[i][j][k][l] += learningrate * error[i][m][n] * padded[j][k * stride + m][l * stride + n];
						}
					}
				}
			}
		}
	}
	updateBias(bias, error);
	return tmp;
}


vector<vector<vector<double>>> calError_in(ParallelCompute pc, vector<vector<vector<double>>> &error, vector<vector<vector<vector<double>>>> &filter,int stride) {
	int I = filter[0].size();
	int L = filter.size();
	int M = filter[0][0].size();
	int N = filter[0][0][0].size();
	int outSize = ((error[0][0].size() - filter[0][0][0].size()) / stride) + 1;
	vector<vector<vector<double>>> tmp;
	resizeVec(tmp,I,outSize,outSize);
	for (int i = 0; i < I; i++) {
		for (int j = 0; j < outSize; j++) {
			for (int k = 0; k < outSize; k++) {
				for (int l = 0; l < L; l++) {
					for (int m = 0; m < M; m++) {
						for (int n = 0; n < N; n++) {
							tmp[i][j][k] += error[l][j * stride + m][k * stride + n] * filter[l][i][M - m - 1][N - n - 1];
						}
					}
				}
			}
		}
	}
	return tmp;
}

void updateFilter(vector<vector<vector<vector<double>>>> &filter, vector<vector<vector<vector<double>>>> &deltaFilter) {
	int I = filter.size();
	int J = filter[0].size();
	int K = filter[0][0].size();
	int L = filter[0][0][0].size();
	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			for (int k = 0; k < K; k++) {
				for (int l = 0; l < L; l++) {
					filter[i][j][k][l] -= deltaFilter[i][j][k][l];
					deltaFilter[i][j][k][l] = 0;
				}
			}
		}
	}
}

vector<vector<vector<double>>> padding(vector<vector<vector<double>>> &pooled, int padsize) {
	int I = pooled.size();
	int J = pooled[0].size();
	int K = pooled[0][0].size();
	vector<vector<vector<double>>> tmp;
	resizeVec(tmp, I, J + (padsize * 2), K + (padsize * 2));
	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			for (int k = 0; k < K; k++) {
				tmp[i][j + padsize][k + padsize] = pooled[i][j][k];
			}
		}
	}
	return tmp;
}

vector<vector<vector<double>>> reluF(vector<vector<vector<double>>> &X) {
	int I = X.size();
	int J = X[0].size();
	int K = X[0][0].size();
	vector<vector<vector<double>>> tmp; 
	resizeVec(tmp , I, J, K);
	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			for (int k = 0; k < K; k++) {
				if (X[i][j][k] > 0)tmp[i][j][k] = X[i][j][k];
				else tmp[i][j][k] = 0;
			}
		}
	}
	return tmp;
}

vector<vector<double>> reluF(vector<vector<double>> &X) {
	int I = X.size();
	int J = X[0].size();
	vector<vector<double>> tmp;
	resizeVec(tmp,I, J);
		for (int i = 0; i < I; i++) {
			for (int j = 0; j < J; j++) {
				if (X[i][j] > 0)tmp[i][j] = X[i][j];
				else tmp[i][j] = 0;
			}
		}
	return tmp;
}

vector<vector<double>> softmax(vector<vector<double>> &X) {
	int I = X[0].size();
	vector<vector<double>> tmp;
	resizeVec(tmp,1, I);
	
	double max = X[0][0];
	for (int i = 0; i < I; i++) {
		if (max < X[0][i]) max = X[0][i];
	}

	double sum = 0;
	for (int i = 0; i < I; i++) {
		double n = X[0][i]-max;
		sum += exp(n);
	}

	for (int i = 0; i < I; i++) {
		double n = X[0][i];
		tmp[0][i] = exp(n) / sum;
	}
	return tmp;
}

vector<vector<double>> dreluF(vector<vector<double>> &X) {
	int I = X.size();
	int J = X[0].size();
	vector<vector<double>> tmp;
	resizeVec(tmp, I, J);
	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			if (X[i][j] > 0) tmp[i][j] = 1;
			else tmp[i][j] = 0;
		}
	}
	return tmp;
}

vector<vector<vector<double>>> dreluF(vector<vector<vector<double>>> &X) {
	int I = X.size();
	int J = X[0].size();
	int K = X[0][0].size();
	vector<vector<vector<double>>> tmp;
	resizeVec(tmp,I, J, K);
	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			for (int k = 0; k < K; k++) {
				if (X[i][j][k] > 0) tmp[i][j][k] = 1;
				else tmp[i][j][k] = 0;
			}
		}
	}
	return tmp;
}


double calcRMSE(vector<vector<double>> &L) {
	int Nrow = L.size();
	int Ncol = L[0].size();

	int TotNum = Nrow * Ncol;

	double mse = 0;
	for (int i = 0; i < Nrow; i++) {
		for (int j = 0; j < Ncol; j++) {
			mse += (L[i][j] * L[i][j]);
		}
	}

	double rmse = sqrt(mse / TotNum);
	return rmse;
}

void resizeVec(vector<vector<double>> &vec2d,int height, int width) {
	vec2d.resize(height,vector<double>(width,0));
}

void resizeVec(vector<vector<vector<double>>> &vec3d, int depth, int height, int width) {
	vec3d.resize(depth,vector<vector<double>>(height, vector<double>(width, 0)));
}

void resizeVec(vector<vector<vector<vector<double>>>> &vec4d, int post_depth, int prev_depth, int height, int width) {
	vec4d.resize(post_depth,vector<vector<vector<double>>>(prev_depth,vector<vector<double>>(height, vector<double>(width, 0))));
}

void setRandom(vector<vector<double>> &vec2d, double rn) {
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<double> distribution(-rn, rn);

	for (int i = 0; i < vec2d.size(); i++) {
		for (int j = 0; j < vec2d[0].size(); j++) {
			vec2d[i][j] = distribution(gen);
		}
	}
}

void setRandom(vector<vector<vector<double>>> &vec3d, double rn) {
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<double> distribution(-rn, rn);

	for (int i = 0; i < vec3d.size(); i++) {
		for (int j = 0; j < vec3d[0].size(); j++) {
			for (int k = 0; k < vec3d[0][0].size(); k++) {

				vec3d[i][j][k] = distribution(gen);
			}
		}
	}
}

void setRandom(vector<vector<vector<vector<double>>>> &vec4d, double rn) {
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<double> distribution(-rn, rn);

	for (int i = 0; i < vec4d.size(); i++) {
		for (int j = 0; j < vec4d[0].size(); j++) {
			for (int k = 0; k < vec4d[0][0].size(); k++) {
				for (int l = 0; l < vec4d[0][0][0].size(); l++) {
					vec4d[i][j][k][l] = distribution(gen);
				}
			}
		}
	}
}

void NormalizeImage(vector<vector<vector<double>>> &img)
{
	for (int i = 0; i < img.size(); i++) {
		for (int j = 0; j < img[0].size(); j++) {
			for (int k = 0; k < img[0][0].size(); k++) {
				img[i][j][k] = img[i][j][k] / 255.f;
			}
		}
	}
}

float* changeTo1D(vector<vector<vector<vector<double>>>>& vec4d)
{
	int post_depth = vec4d.size();
	int prev_depth = vec4d[0].size();
	int height = vec4d[0][0].size();
	int width = vec4d[0][0][0].size();

	float* tmp = new float[prev_depth*post_depth*height*width];
	for (int i = 0; i < prev_depth; i++) {
		for (int j = 0; j < post_depth; j++) {
			for (int k = 0; k < height; k++) {
				for (int l = 0; l < width; l++) {
					tmp[i*post_depth*height*width + j*height*width + k*width + l] = vec4d[j][i][k][l];
				}
			}
		}
	}
	return tmp;
}
float* changeTo1D(vector<vector<vector<double>>>& vec3d)
{
	int depth = vec3d.size();
	int height = vec3d[0].size();
	int width = vec3d[0][0].size();

	float* tmp = new float[depth*height*width];
	for (int i = 0; i < depth; i++) {
		for (int j = 0; j < height; j++) {
			for (int k = 0; k < width; k++) {
				tmp[i*height*width + j*width + k] = vec3d[i][j][k];
			}
		}
	}
	return tmp;
}
float* changeTo1D(vector<vector<double>>& vec2d)
{
	int height = vec2d.size();
	int width = vec2d[0].size();

	float* tmp = new float[height*width];
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			tmp[i*width + j] = vec2d[i][j];
		}
	}
	return tmp;
}

void changeTo2D(vector<vector<double>>& vec2d, float* ptr2d)
{
	int height = vec2d.size();
	int width = vec2d[0].size();

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			vec2d[i][j] = ptr2d[i*width + j];
		}
	}
}
void changeTo3D(vector<vector<vector<double>>>& vec3d, float* ptr3d)
{
	int depth = vec3d.size();
	int height = vec3d[0].size();
	int width = vec3d[0][0].size();

	for (int i = 0; i < depth; i++) {
		for (int j = 0; j < height; j++) {
			for (int k = 0; k < width; k++) {
				vec3d[i][j][k] = ptr3d[i*height*width + j*width + k];
			}
		}
	}
}
void changeTo4D(vector<vector<vector<vector<double>>>>& vec4d, float* ptr4d)
{
	int post_depth = vec4d.size();
	int prev_depth = vec4d[0].size();
	int height = vec4d[0][0].size();
	int width = vec4d[0][0][0].size();

	for (int i = 0; i < prev_depth; i++) {
		for (int j = 0; j < post_depth; j++) {
			for (int k = 0; k < height; k++) {
				for (int l = 0; l < width; l++) {
					vec4d[j][i][k][l] = ptr4d[i*post_depth*height*width + j*height*width + k*width + l];
				}
			}
		}
	}
}

void GDMoment(vector<vector<vector<vector<double>>>> &deltaFilter, vector<vector<vector<double>>> &deltaBias, vector<vector<vector<vector<double>>>> opt_filter, vector<vector<vector<double>>> &opt_bias, vector<vector<vector<vector<double>>>> &gradientFilter, vector<vector<vector<double>>> &error,  double momentum, int ep) 
{
	int I = gradientFilter.size();
	int J = gradientFilter[0].size();
	int K = gradientFilter[0][0].size();
	int L = gradientFilter[0][0][0].size();
	int M = error[0].size();
	int N = error[0][0].size();

	double learningrate = 0.001;
	if (ep == 0) {
		resizeVec(opt_filter, I, J, K, L);
		resizeVec(opt_bias, I, M, N);
	}

	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			for (int k = 0; k < K; k++) {
				for (int l = 0; l < L; l++) {
					opt_filter[i][j][k][l] = momentum * opt_filter[i][j][k][l] + learningrate * gradientFilter[i][j][k][l];
					deltaFilter[i][j][k][l] = opt_filter[i][j][k][l];
				}
			}
		}
	}

	for (int i = 0; i < J; i++) {
		for (int m = 0; m < M; m++) {
			for (int n = 0; n < N; n++) {
				opt_bias[i][m][n] = momentum * opt_bias[i][m][n] + learningrate * error[i][m][n];
				deltaBias[i][m][n] = opt_bias[i][m][n];
			}
		}
	}
}

void RMSProp(vector<vector<vector<vector<double>>>> &deltaFilter, vector<vector<vector<double>>> &deltaBias, vector<vector<vector<vector<double>>>> &opt_filter, vector<vector<vector<double>>> &opt_bias, vector<vector<vector<vector<double>>>> &gradientFilter, vector<vector<vector<double>>> &error, double gamma, int ep) 
{
	int I = gradientFilter.size();
	int J = gradientFilter[0].size();
	int K = gradientFilter[0][0].size();
	int L = gradientFilter[0][0][0].size();
	int M = error[0].size();
	int N = error[0][0].size();

	double learningrate = 0.001;
	if (ep == 0) {
		resizeVec(opt_filter, I, J, K, L);
		resizeVec(opt_bias, I, M, N);
	}

	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			for (int k = 0; k < K; k++) {
				for (int l = 0; l < L; l++) {
					opt_filter[i][j][k][l] = (gamma * opt_filter[i][j][k][l]) + (1 - gamma)* (gradientFilter[i][j][k][l] * gradientFilter[i][j][k][l]);
					deltaFilter[i][j][k][l] = learningrate * gradientFilter[i][j][k][l] / sqrt(opt_filter[i][j][k][l] + 10e-8);
				}
			}
		}
	}

	for (int i = 0; i < I; i++) {
		for (int m = 0; m < M; m++) {
			for (int n = 0; n < N; n++) {
				opt_bias[i][m][n] = (gamma * opt_bias[i][m][n]) + (1 - gamma)* (error[i][m][n] * error[i][m][n]);
				deltaBias[i][m][n] = learningrate * error[i][m][n] / sqrt(opt_bias[i][m][n] + 10e-8);
			}
		}
	}
}

void Adagrad(vector<vector<vector<vector<double>>>> &deltaFilter, vector<vector<vector<double>>> &deltaBias, vector<vector<vector<vector<double>>>> &opt_filter, vector<vector<vector<double>>> &opt_bias, vector<vector<vector<vector<double>>>> &gradientFilter, vector<vector<vector<double>>> &error, int ep) 
{
	int I = gradientFilter.size();
	int J = gradientFilter[0].size();
	int K = gradientFilter[0][0].size();
	int L = gradientFilter[0][0][0].size();
	int M = error[0].size();
	int N = error[0][0].size();

	double learningrate = 0.001;
	if (ep == 0) {
		resizeVec(opt_filter, I, J, K, L);
		resizeVec(opt_bias, I, M, N);
	}

	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			for (int k = 0; k < K; k++) {
				for (int l = 0; l < L; l++) {
					opt_filter[i][j][k][l] = opt_filter[i][j][k][l] + gradientFilter[i][j][k][l] * gradientFilter[i][j][k][l];
					deltaFilter[i][j][k][l] = learningrate * gradientFilter[i][j][k][l] / sqrt(opt_filter[i][j][k][l] + 10e-8);
				}
			}
		}
	}

	for (int i = 0; i < I; i++) {
		for (int m = 0; m < M; m++) {
			for (int n = 0; n < N; n++) {
				opt_bias[i][m][n] =  opt_bias[i][m][n] + error[i][m][n] * error[i][m][n];
				deltaBias[i][m][n] = learningrate * error[i][m][n] / sqrt(opt_bias[i][m][n] + 10e-8);
			}
		}
	}
}

void Adam(vector<vector<vector<vector<double>>>> &deltaFilter, vector<vector<vector<double>>> &deltaBias, vector<vector<vector<vector<double>>>> opt_filter, vector<vector<vector<double>>> &opt_bias, vector<vector<vector<vector<double>>>> mom_filter, vector<vector<vector<double>>> &mom_bias, vector<vector<vector<vector<double>>>> gradientFilter, vector<vector<vector<double>>> &error, double beta1, double beta2, int ep) 
{
	int I = gradientFilter.size();
	int J = gradientFilter[0].size();
	int K = gradientFilter[0][0].size();
	int L = gradientFilter[0][0][0].size();
	int M = error[0].size();
	int N = error[0][0].size();
	double learningrate = 0.001;
	if (ep == 0) {
		resizeVec(opt_filter, I, J, K, L);
		resizeVec(mom_filter, I, J, K, L);
		resizeVec(opt_bias, I, M, N);
		resizeVec(mom_bias, I, M, N);
	}

	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			for (int k = 0; k < K; k++) {
				for (int l = 0; l < L; l++) {
					mom_filter[i][j][k][l] = (beta1 * mom_filter[i][j][k][l]) + (1 - beta1) * gradientFilter[i][j][k][l];
					opt_filter[i][j][k][l] = (beta2 * opt_filter[i][j][k][l]) + (1 - beta2) * (gradientFilter[i][j][k][l] * gradientFilter[i][j][k][l]);
					deltaFilter[i][j][k][l] = mom_filter[i][j][k][l] * (learningrate / sqrt(opt_filter[i][j][k][l] + 1e-8));
				}
			}
		}
	}

	for (int i = 0; i < I; i++) {
		for (int m = 0; m < M; m++) {
			for (int n = 0; n < N; n++) {
				mom_bias[i][m][n] = beta1 * mom_bias[i][m][n] + (1 - beta1) * error[i][m][n];
				opt_bias[i][m][n] = beta2 * opt_bias[i][m][n] + (1 - beta2) * (error[i][m][n] * error[i][m][n]);
				deltaBias[i][m][n] = mom_bias[i][m][n] * (learningrate / sqrt(opt_bias[i][m][n] + 1e-8));
			}
		}
	}
}

void RMSProp(vector<vector<double>> &deltaWeight, vector<vector<double>> &deltaBias, vector<vector<double>> opt_weight, vector<vector<double>> &opt_bias, vector<vector<double>> gradientWeight, vector<vector<double>> &error, double gamma, int ep)
{
	int I = gradientWeight.size();
	int J = gradientWeight[0].size();

	int M = error.size();
	int N = error[0].size();

	double learningrate = 0.001;
	if (ep == 0) {
		resizeVec(opt_weight, I, J);
		resizeVec(opt_bias, M, N);
	}

	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			opt_weight[i][j] = gamma * opt_weight[i][j] + (1 - gamma) * (gradientWeight[i][j] * gradientWeight[i][j]);
			deltaWeight[i][j] = learningrate * gradientWeight[i][j] / (sqrt(opt_weight[i][j] + 10e-8));
		}
	}

	for (int m = 0; m < M; m++) {
		for (int n = 0; n < N; n++) {
			opt_bias[m][n] = gamma * opt_bias[m][n] + (1 - gamma) * (error[m][n] * error[m][n]);
			deltaBias[m][n] = learningrate * error[m][n] / (sqrt(opt_bias[m][n] + 10e-8));
		}
	}
}

void GDMoment(vector<vector<double>> &deltaWeight, vector<vector<double>> &deltaBias, vector<vector<double>> opt_weight, vector<vector<double>> &opt_bias, vector<vector<double>> gradientWeight, vector<vector<double>> &error, double momentum, int ep)
{
	int I = gradientWeight.size();
	int J = gradientWeight[0].size();

	int M = error.size();
	int N = error[0].size();

	double learningrate = 0.001;
	if (ep == 0) {
		resizeVec(opt_weight, I, J);
		resizeVec(opt_bias, M, N);
	}

	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			opt_weight[i][j] = momentum * opt_weight[i][j] + learningrate * gradientWeight[i][j];
			deltaWeight[i][j] = opt_weight[i][j];
		}
	}

	for (int m = 0; m < M; m++) {
		for (int n = 0; n < N; n++) {
			opt_bias[m][n] = momentum * opt_bias[m][n] + learningrate* error[m][n];
			deltaBias[m][n] = opt_bias[m][n];
		}
	}
}

void Adagrad(vector<vector<double>> &deltaWeight, vector<vector<double>> &deltaBias, vector<vector<double>> opt_weight, vector<vector<double>> &opt_bias, vector<vector<double>> gradientWeight, vector<vector<double>> &error, int ep)
{
	int I = gradientWeight.size();
	int J = gradientWeight[0].size();

	int M = error.size();
	int N = error[0].size();

	double learningrate = 0.001;
	if (ep == 0) {
		resizeVec(opt_weight, I, J);
		resizeVec(opt_bias, M, N);
	}

	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			opt_weight[i][j] = opt_weight[i][j] + (gradientWeight[i][j] * gradientWeight[i][j]);
			deltaWeight[i][j] = learningrate * gradientWeight[i][j] / (sqrt(opt_weight[i][j] + 10e-8));
		}
	}

	for (int m = 0; m < M; m++) {
		for (int n = 0; n < N; n++) {
			opt_bias[m][n] = opt_bias[m][n] + (error[m][n] * error[m][n]);
			deltaBias[m][n] = learningrate * error[m][n] / (sqrt(opt_bias[m][n] + 10e-8));
		}
	}
}

void Adam(vector<vector<double>> &deltaWeight, vector<vector<double>> &deltaBias, vector<vector<double>> opt_weight, vector<vector<double>> &opt_bias, vector<vector<double>> mom_weight, vector<vector<double>> &mom_bias, vector<vector<double>> gradientWeight, vector<vector<double>> &error, double beta1, double beta2, int ep)
{
	int I = gradientWeight.size();
	int J = gradientWeight[0].size();

	int M = error.size();
	int N = error[0].size();

	double learningrate = 0.001;
	if (ep == 0) {
		resizeVec(opt_weight, I, J);
		resizeVec(mom_weight, I, J);
		resizeVec(opt_bias, M, N);
		resizeVec(mom_bias, M, N);
	}

	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			mom_weight[i][j] = beta1 * mom_weight[i][j] + (1 - beta1) * gradientWeight[i][j];
			opt_weight[i][j] = beta2 * opt_weight[i][j] + (1 - beta2) * (gradientWeight[i][j] * gradientWeight[i][j]);
			deltaWeight[i][j] = mom_weight[i][j] * (learningrate / (sqrt(opt_weight[i][j] + 10e-8)));
		}
	}

	for (int m = 0; m < M; m++) {
		for (int n = 0; n < N; n++) {
			mom_bias[m][n] = beta1 * mom_bias[m][n] + (1 - beta1) * error[m][n];
			opt_bias[m][n] = beta2 * opt_bias[m][n] + (1 - beta2) * (error[m][n] * error[m][n]);
			deltaBias[m][n] = mom_bias[m][n] * (learningrate / (sqrt(opt_bias[m][n] + 10e-8)));
		}
	}
}

vector<vector<vector<double>>> bNormalize(vector<vector<vector<double>>> X)
{
	vector<vector<vector<double>>> Y;
	int I = X.size();
	int J = X[0].size();
	int K = X[0][0].size();
	double sum = 0;
	

	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			for (int k = 0; k < K; k++) {
				sum += X[i][j][k];
			}
		}
	}

	double mean = sum / I*J*K;
	double var = 0;
	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			for (int k = 0; k < K; k++) {
				var += (X[i][j][k] - mean)*(X[i][j][k] - mean) / I*J*K;
			}
		}
	}

	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			for (int k = 0; k < K; k++) {
				Y[i][j][k] = (X[i][j][k] - mean) / (sqrt(var + 1e-8));
			}
		}
	}
	
	return Y;
}
