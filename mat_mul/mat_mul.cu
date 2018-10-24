#include <iostream>

using namespace std;

__global__ void Mat_Mul_Global (float *A, float *B, float *C, int width){

  int i = blockIdx.y * blockDim.y + threadIdx.y;  //use 2D thread-blocks for convinience, use 1D is also OK. 
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if ((i < width) && (j <width)){  // usually the number of threads is larger than the array size
     float value = 0.;   // Non array variable in kenel is saved in the register of every thread
     for(int k=0; k<width; k++){   // every thread compute this loop of k
        value += A[i*width + k] * B[k*width + j]; // the scalar value is saved in the register for every thread (not in global memory), so there is no data race problem.
     }
     C[i*width+j] = value;  // assign to array C on global memory
  }

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

  int block_width = 16;
  int grid_width = ceil(n/block_width);
  // dim3 is a CUDA built-in struct
  dim3 dim_grid(grid_width, grid_width);  // num of blocks = (int(n/16) + 1)^2, efficient if > 6.
  dim3 dim_block(block_width, block_width);  // number of theads per block = 16*16 =256
  Mat_Mul_Global<<<dim_grid, dim_block>>> (d_A, d_B, d_C, n);  // lauch kenel

  cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

  for(int i=0; i<n; i++)
     for(int j=0; j<n; j++){
     cout<<i<<"  "<<j<<"  "<<C[i*n+j]<<endl;
  }

  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
  free(A); free(B); free(C);
 
}

