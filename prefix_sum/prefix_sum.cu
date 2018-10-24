#include<iostream>
#define SECTION_SIZE 32

using namespace std;

__global__ void Prefix_sum_oneblock_kernel(float *X, float *Y, int InputSize, float *S) {

   __shared__ float XY[SECTION_SIZE];

   int i = blockIdx.x*blockDim.x+ threadIdx.x;

   XY[threadIdx.x] = X[i];
   /*if (i < InputSize && threadIdx.x != 0) {
      XY[threadIdx.x] = X[i-1];
   }else{
      XY[threadIdx.x] = 0;
   }*/

// the code below performs iterative scan on XY
   for (unsigned int stride = 1; stride <= threadIdx.x; stride *= 2) {
       __syncthreads();
       XY[threadIdx.x] += XY[threadIdx.x-stride];
   }

   if(i<InputSize) Y[i] = XY[threadIdx.x];
   //Y[i] = XY[threadIdx.x];

   __syncthreads();
   if(threadIdx.x == 0) S[blockIdx.x] = XY[SECTION_SIZE-1];  // get the last element in each section

}

__global__ void Add_scalar_to_subvector(float *Y, float *S, int InputSize){
   
   int i = (blockIdx.x+1)*blockDim.x+ threadIdx.x;
   if(i<InputSize) Y[i] += S[blockIdx.x];
   //Y[i] += S[blockIdx.x];

}


int main(){

  int n;
  cin >> n;
  int size = n*sizeof(float);

  //float *A, *B, *C;
  float *X_h = (float *) malloc( size );  // allocate CPU memory
  float *Y_h = (float *) malloc( size );

  for(int i=0; i<n; i++){ X_h[i] = 1; Y_h[i]=0; }

  float *X, *Y, *S, *S1;
  cudaMalloc(&X, size);  // allocate GPU memory
  cudaMalloc(&Y, size);
  cudaMemcpy(X, X_h, size, cudaMemcpyHostToDevice);

  int BLOCK_SIZE = SECTION_SIZE;
  int GRID_SIZE=ceil(n/BLOCK_SIZE);
  int size_s = GRID_SIZE*sizeof(float);
  cudaMalloc(&S, size_s);
  cudaMalloc(&S1, size_s);
  Prefix_sum_oneblock_kernel<<<GRID_SIZE,BLOCK_SIZE>>> (X, Y, n, S);
  Prefix_sum_oneblock_kernel<<<GRID_SIZE,BLOCK_SIZE>>> (S, S, n, S1);
  Add_scalar_to_subvector<<<GRID_SIZE,BLOCK_SIZE>>> (Y, S, n);

  cudaMemcpy(Y_h, Y, size, cudaMemcpyDeviceToHost);

   for(int i=0; i<n; i++){
     cout<<i<<"  "<<Y_h[i]<<endl;
  }

  cudaFree(X); cudaFree(Y);
  free(X_h); free(Y_h);

}
