#include<iostream>
#define tile_width 16

using namespace std;

__global__ void Mat_Mul_Shared(float *d_A, float *d_B, float *d_C, int width) {

   int i = blockIdx.y * blockDim.y + threadIdx.y;
   int j = blockIdx.x * blockDim.x + threadIdx.x;

   __shared__ float M[tile_width][tile_width];
   __shared__ float N[tile_width][tile_width];

   int n_tile = width / tile_width;  //number of tiles
   float value = 0;

   for(int m=0; m < n_tile; m++ ){  // m loop of tiles
       // copy from globalb memory to shared memory       
       M[threadIdx.y][threadIdx.x] = d_A[i*width + m*tile_width + threadIdx.x];
       N[threadIdx.y][threadIdx.x] = d_B[(m*tile_width + threadIdx.y)*width + j];
       __syncthreads();  // wait all threads finish copying

      for(int k=0; k < tile_width; k++){  // k loop within a tile, in the m loop of tiles: together loop over width to perfomr the dot product of a row of d_A  and a coloumn of d_B
          value += M[threadIdx.y][k] * N[k][threadIdx.x];  // scalar value is in the register of a thread
      }
      __syncthreads();  // wait ll threads finish partial dot product
   }

   d_C[i*width + j] = value;  // assign value in register of every thread to corresponding elements of d_C in global memory

}


int main(){

  int n;
  cin >> n;
  int size = n*n*sizeof(float);

  //float *A, *B, *C;
  float *A = (float *) malloc( size );  // allocate CPU memory
  float *B = (float *) malloc( size );
  float *C = (float *) malloc( size );

  for(int i=0; i<n; i++)
     for(int j=0; j<n; j++){
         int idx = i*n + j;
         A[idx] = 1.;
         B[idx] = 2.;
         C[idx] = 0.;
  }

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, size);  // allocate GPU memory
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);
  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

  int block_width = tile_width;
  int grid_width = ceil(n/block_width);
  // dim3 is a CUDA built-in struct
  dim3 dim_grid(grid_width, grid_width);  // num of blocks = (int(n/16) + 1)^2, efficient if > 6.
  dim3 dim_block(block_width, block_width);  // number of theads per block = 16*16 =256

  Mat_Mul_Shared<<<dim_grid, dim_block>>> (d_A, d_B, d_C, n);  // lauch kenel

  cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

  for(int i=0; i<n; i++)
     for(int j=0; j<n; j++){
     cout<<i<<"  "<<j<<"  "<<C[i*n+j]<<endl;
  } 

  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
  free(A); free(B); free(C);



}
