#pragma once
#include "opencv2/opencv.hpp"
#include "computeFunctions.h"
#include <algorithm>
#include <experimental/filesystem>
#include <vector>
#include <math.h>
#include <random>
#include <chrono> 
#include <io.h>

using namespace std;
using namespace cv;


vector<cv::String> getImage(string& path);
vector<vector<vector<double>>> setImage(string& path);
vector<vector<double>> setTarget(string& imagePath);
vector<vector<vector<double>>> convolution(ParallelCompute pc, vector<vector<vector<double>>>& X, vector<vector<vector<double>>>& B, vector<vector<vector<vector<double>>>>& F, int stride);
vector<vector<vector<double>>> maxPooling(vector<vector<vector<double>>>& activated, vector<vector<vector<double>>> &position1, vector<vector<vector<double>>> &position2, int poolsize, int stride);
vector<vector<double>> flatten(vector<vector<vector<double>>>& input);
vector<vector<double>> loss(vector<vector<double>> &output, vector<vector<double>>& target);
vector<vector<double>> calError(vector<vector<double>> &errorIn, vector<vector<double>> &input);
vector<vector<double>> calWeightDiff(vector<vector<double>> &bias, vector<vector<double>> &error, vector<vector<double>> &conv);
vector<vector<double>> calError_in(ParallelCompute pc, vector<vector<double>> &error, vector<vector<double>> &weight);
vector<vector<vector<double>>> dflatten(vector<vector<double>>& errorIn, int width);
vector<vector<vector<double>>> calError(vector<vector<vector<double>>> &errorIn, vector<vector<vector<double>>> &input);
vector<vector<vector<double>>> poolError(vector<vector<vector<double>>> &error, vector<vector<vector<double>>> &position1, vector<vector<vector<double>>> &position2, int stride, int num);
vector<vector<vector<vector<double>>>> calFilterDiff(vector<vector<vector<vector<double>>>> &filter, vector<vector<vector<double>>> &bias, vector<vector<vector<double>>> &error, vector<vector<vector<double>>> &padded, int stride);
vector<vector<vector<double>>> calError_in(ParallelCompute pc, vector<vector<vector<double>>> &error, vector<vector<vector<vector<double>>>> &filter, int stride);
vector<vector<vector<double>>> padding(vector<vector<vector<double>>>&pooled, int padsize);
vector<vector<vector<double>>> reluF(vector<vector<vector<double>>>& X);
vector<vector<double>> reluF(vector<vector<double>>& X);
vector<vector<double>> softmax(vector<vector<double>>& X);
vector<vector<double>> dreluF(vector<vector<double>>& X);
vector<vector<vector<double>>> dreluF(vector<vector<vector<double>>>& X);
void multiplication(ParallelCompute pc, vector<vector<double>> &input, vector<vector<double>> &output, vector<vector<double>> &bias, vector<vector<double>> &weight);
void updateBias2(vector<vector<double>>& bias, vector<vector<double>>& error);
void updateBias(vector<vector<vector<double>>> &bias, vector<vector<vector<double>>> &error);
void updateWeight(vector<vector<double>> &weight, vector<vector<double>> & deltaWeight);
void updateFilter(vector<vector<vector<vector<double>>>> &filter, vector<vector<vector<vector<double>>>> &deltaFilter);
void resizeVec(vector<vector<double>>& vec2d, int height, int width);
void resizeVec(vector<vector<vector<double>>>& vec3d, int depth, int height, int width);
void resizeVec(vector<vector<vector<vector<double>>>>& vec4d, int post_depth, int prev_depth, int height, int width);
void setRandom(vector<vector<double>>& vec2d, double rn);
void setRandom(vector<vector<vector<double>>>& vec3d, double rn);
void setRandom(vector<vector<vector<vector<double>>>>& vec4d, double rn);
void NormalizeImage(vector<vector<vector<double>>>& img);
double calcRMSE(vector<vector<double>> &L);

float* changeTo1D(vector<vector<vector<vector<double>>>>& vec4d);
float* changeTo1D(vector<vector<vector<double>>>& vec3d);
float* changeTo1D(vector<vector<double>>& vec2d);

void changeTo2D(vector<vector<double>>& vec2d, float* ptr2d);
void changeTo3D(vector<vector<vector<double>>>& vec3d, float* ptr3d);
void changeTo4D(vector<vector<vector<vector<double>>>>& vec4d, float* ptr4d);

void GDMoment(vector<vector<vector<vector<double>>>> &deltaFilter, vector<vector<vector<double>>> &deltaBias, vector<vector<vector<vector<double>>>> opt_filter, vector<vector<vector<double>>> &opt_bias, vector<vector<vector<vector<double>>>> &gradientFilter, vector<vector<vector<double>>> &error, double momentum, int ep);
void RMSProp(vector<vector<vector<vector<double>>>> &deltaFilter, vector<vector<vector<double>>> &deltaBias, vector<vector<vector<vector<double>>>> &opt_filter, vector<vector<vector<double>>> &opt_bias, vector<vector<vector<vector<double>>>> &gradientFilter, vector<vector<vector<double>>> &error, double gamma, int ep);
void Adagrad(vector<vector<vector<vector<double>>>> &deltaFilter, vector<vector<vector<double>>> &deltaBias, vector<vector<vector<vector<double>>>> &opt_filter, vector<vector<vector<double>>> &opt_bias, vector<vector<vector<vector<double>>>> &gradientFilter, vector<vector<vector<double>>> &error, int ep);
void Adam(vector<vector<vector<vector<double>>>> &deltaFilter, vector<vector<vector<double>>> &deltaBias, vector<vector<vector<vector<double>>>> opt_filter, vector<vector<vector<double>>> &opt_bias, vector<vector<vector<vector<double>>>> mom_filter, vector<vector<vector<double>>> &mom_bias, vector<vector<vector<vector<double>>>> gradientFilter, vector<vector<vector<double>>> &error, double beta1, double beta2, int ep);

void RMSProp(vector<vector<double>> &deltaWeight, vector<vector<double>> &deltaBias, vector<vector<double>> opt_weight, vector<vector<double>> &opt_bias, vector<vector<double>> gradientWeight, vector<vector<double>> &error, double gamma, int ep);
void GDMoment(vector<vector<double>> &deltaWeight, vector<vector<double>> &deltaBias, vector<vector<double>> opt_weight, vector<vector<double>> &opt_bias, vector<vector<double>> gradientWeight, vector<vector<double>> &error, double momentum, int ep);
void Adagrad(vector<vector<double>> &deltaWeight, vector<vector<double>> &deltaBias, vector<vector<double>> opt_weight, vector<vector<double>> &opt_bias, vector<vector<double>> gradientWeight, vector<vector<double>> &error, int ep);
void Adam(vector<vector<double>> &deltaWeight, vector<vector<double>> &deltaBias, vector<vector<double>> opt_weight, vector<vector<double>> &opt_bias, vector<vector<double>> mom_weight, vector<vector<double>> &mom_bias, vector<vector<double>> gradientWeight, vector<vector<double>> &error, double beta1, double beta2, int ep);
vector<vector<vector<double>>> bNormalize(vector<vector<vector<double>>> X);
