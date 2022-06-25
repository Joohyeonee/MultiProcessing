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

__kernel void convolution(__global float *X,__global float *F,__global float *Y, __global int *size){
        int i = get_global_id(1);
        int j = get_global_id(0);
		float sum = 0.0f;

        if(i < size[3] * size[9] && j < size[4] * size[5]){
         for(int k = (i/size[3])*size[0] ; k < (i/size[3] + 1)*size[0] ; k++){
            for(int p = 0 ; p < size[6] ; p++){
               for(int q = 0 ; q < size[7] ; q++){
                  sum += X[k*size[1]*size[2] + (size[8]*(j/size[5]) + p)*size[2] + (size[8]*(j%size[5]) + q)] * F[k*size[3]*size[6]*size[7] + (i%size[3])*size[6]*size[7] + p*size[7] + q];
               }
            }
         }
         Y[i*size[4]*size[5] + j] = sum;
        }
}
