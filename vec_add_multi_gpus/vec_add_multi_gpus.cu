#include <stdio.h>
#include <iostream>

using namespace std;

__global__ void vec_add(float *A, float *B, float *C, int n_array){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx<n_array){
      C[idx] = A[idx] + B[idx];
    }

}


int main(){

    int ngpus=2;
    cudaGetDeviceCount(&ngpus);
    printf("Number of GPUs: %i\n", ngpus);

    float *d_A[ngpus], *d_B[ngpus], *d_C[ngpus];
    float *h_A[ngpus], *h_B[ngpus], *h_C[ngpus];
    cudaStream_t   stream[ngpus];

    int n=1024;
    cin >> n ;
    int n_d = n/ngpus; // number of elements on one device
    //int size_d = n_d * sizeof(float);
    size_t size_d = n_d * sizeof(float);
   
    int block_size = 256; //32;  //256;
    int grid_size = n_d / block_size;

    #pragma omp parallel for
    for(int i=0; i<ngpus; i++){

       cudaSetDevice(i);  // set current device

       cudaMalloc(&d_A[i], size_d);  // allocate GPU memory
       cudaMalloc(&d_B[i], size_d);
       cudaMalloc(&d_C[i], size_d);
       
       cudaMallocHost(&h_A[i], size_d);  //allocate CPU memory for each section corresponding to each GPU
       cudaMallocHost(&h_B[i], size_d);
       cudaMallocHost(&h_C[i], size_d);

       cudaStreamCreate(&stream[i]);

       for(int j=0; j<n_d; j++){ // initialize data on CPU
          h_A[i][j] = 2;
          h_B[i][j] = 3;
       }

       cudaMemcpyAsync(d_A[i], h_A[i], size_d, cudaMemcpyHostToDevice, stream[i]);
       cudaMemcpyAsync(d_B[i], h_B[i], size_d, cudaMemcpyHostToDevice, stream[i]);

       vec_add<<<grid_size, block_size, 0, stream[i]>>>(d_A[i], d_B[i], d_C[i], n_d); // what is the 3rd arg?

       cudaMemcpyAsync(h_C[i], d_C[i], size_d, cudaMemcpyDeviceToHost, stream[i]);
 
    }
   
    cudaDeviceSynchronize();

    /*for(int i=1; i<ngpus; i++)
       for(int j=1; j<n_d; j++){
       printf("%f\n", h_C[i][j]);
    }*/

}
