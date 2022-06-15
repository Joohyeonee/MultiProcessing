#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

__kernel void matMul(__global float *A,__global float *B,__global float *Y, __global int *size){
        int i = get_global_id(1);
        int j = get_global_id(0);
        float sum = 0.0f;

        if(i<size[0] && j <size[3]){
                for(int k=0; k<size[1]; k++){
                        sum+=A[i * size[1] + k] * B[k * size[3] + j];
                }
                Y[i * size[3] + j] = sum;
        }
}

__kernel void matMulwithBias(__global float *A,__global float *B, __global float *B0, __global float *Y, __global int *size){
        int i = get_global_id(1);
        int j = get_global_id(0);
        float sum = 0.0f;

        if(i<size[0] && j <size[3]){
                for(int k=0; k<size[1]; k++){
                        sum+=A[i * size[1] + k] * B[k * size[3] + j];
                }
                Y[i * size[3] + j] = sum + B0[i * size[3] + j];
        }
}

__kernel void convolution2D(__global float *X,__global float *F,__global float *Y, __global int *size){
        int i = get_global_id(1);
        int j = get_global_id(0);
		float sum = 0.0f;

        if(i < size[2] && j < size[3]){
			for (int m = 0; m < size[4]; m++) {
				for (int n = 0; n < size[5]; n++) {
					sum += X[(size[6]*i + m)*size[1] + (size[6]*j + n)] * F[m*size[5] + n];
				}
			}
         Y[i*size[3] + j] = sum;
        }
}

__kernel void convolution3D(__global float *X,__global float *F,__global float *Y, __global int *size){
        int i = get_global_id(1);
        int j = get_global_id(0);
		float sum = 0.0f;

        if(i < size[3] * size[9] && j < size[4] * size[5]){
         for(int k = (i/size[3])*size[0] ; k < (i/size[3] + 1)*size[0] ; k++){
            for(int m = 0 ; m < size[6] ; m++){
               for(int n = 0 ; n < size[7] ; n++){
                  sum += X[k*size[1]*size[2] + (size[8]*(j/size[5]) + m)*size[2] + (size[8]*(j%size[5]) + n)] * F[k*size[3]*size[6]*size[7] + (i%size[3])*size[6]*size[7] + m*size[7] + n];
               }
            }
         }
         Y[i*size[4]*size[5] + j] = sum;
        }
}

__kernel void convolution3DwithBias(__global float *X,__global float *F, __global float *B0, __global float *Y, __global int *size){
        int i = get_global_id(1);
        int j = get_global_id(0);
		float sum = 0.0f;

        if(i < size[3] * size[9] && j < size[4] * size[5]){
          for(int k = (i/size[3])*size[0] ; k < (i/size[3] + 1)*size[0] ; k++){
            for(int m = 0 ; m < size[6] ; m++){
               for(int n = 0 ; n < size[7] ; n++){
                  sum += X[k*size[1]*size[2] + (size[8]*(j/size[5]) + m)*size[2] + (size[8]*(j%size[5]) + n)] * F[k*size[3]*size[6]*size[7] + (i%size[3])*size[6]*size[7] + m*size[7] + n];
               }
            }
         }
         Y[i*size[4]*size[5] + j] = sum + B0[i*size[4]*size[5] + j];
       }
}

