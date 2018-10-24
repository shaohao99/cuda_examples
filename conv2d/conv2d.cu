#include<iostream>

using namespace std;

#define mask_width 16
#define tile_width 16
__constant__ float M_dc[mask_width][mask_width];  // save mask in constant memory


__global__ void conv2d_kernel(float *N_d, float *P_d, int n) {

   int i = blockIdx.y * blockDim.y + threadIdx.y;  // link local thread idx to global array idx
   int j = blockIdx.x * blockDim.x + threadIdx.x;

   __shared__ float K_ds[tile_width][tile_width];  // in shared mmemory of every block

   K_ds[threadIdx.y][threadIdx.x] = N_d[i*n + j];  // copy from global memory to shared memory
   __syncthreads();

   float value=0;  // in register of every thread

   int half_mask_width = mask_width/2;
   int i_global_start = i - half_mask_width;  // global start index within a mask
   int j_global_start = j - half_mask_width;
   int i_local_start =  threadIdx.y - half_mask_width;  // local start index within a mask
   int j_local_start =  threadIdx.x - half_mask_width;  

   int i_block_start = blockIdx.y * blockDim.y;   // global index of the boundary of the current block
   int i_block_end = (blockIdx.y + 1)* blockDim.y - 1;
   int j_block_start = blockIdx.x * blockDim.x;
   int j_block_end = (blockIdx.x + 1)* blockDim.x - 1;

   if( i<n && j<n ){

     for(int ii=0; ii<mask_width; ii++){   // loops of the mask
       int i_global = i_global_start + ii;
       int i_local = i_local_start + ii;
       for(int jj=0; jj<mask_width; jj++){
         int j_global = j_global_start + jj;
         int j_local = j_local_start + jj;

         if( i_global >= 0 && i_global < n && j_global >= 0 && j_global < n){
             if( i_global >= i_block_start && i_global <= i_block_end && j_global >= j_block_start && j_global <= j_block_end ){   // for points in the tile/block
                value += K_ds[i_local][j_local] * M_dc[ii][jj];  // sub matrix from shared memory
             }else{  // for points out of the tile/block
                value += N_d[i_global*n + j_global] * M_dc[ii][jj];  // sub matrix from L2 cache or global memory
             }
         }  // end if global index in range
       } /// end jj
     }  // end ii

     P_d[i*n+j] = value; // output value of i,j to global memory

   } // end if index in array range

}  // end conv2d_kernel


// =========  main program  =============
int main(){

   int n=32;
   cin>>n;
   int size = n*n*sizeof(float);
   int mask_size = mask_width*mask_width*sizeof(float);

// data on cpu 
   float *N=(float *) malloc(size);
   float *P=(float *) malloc(size);
   //float *M=(float *) malloc(size_mask);
   float M[mask_width][mask_width];

   for(int i=0; i<n; i++)
     for(int j=0; j<n; j++){
         N[i*n + j] = 2.;
   }

   for(int i=0; i<mask_width; i++)
     for(int j=0; j<mask_width; j++){
         M[i][j] = 3.;
   }

// copy to gpu
   float *N_d, *P_d;   // gpu global memory
   cudaMalloc(&N_d, size);
   cudaMalloc(&P_d, size);

   cudaMemcpy(N_d, N, size, cudaMemcpyHostToDevice);
   cudaMemcpyToSymbol(M_dc, M, mask_size); 

// launch kernel
   int grid_width = ceil(n/tile_width);
   dim3 dim_grid(grid_width, grid_width);
   dim3 dim_block(tile_width, tile_width);
   conv2d_kernel<<< dim_grid, dim_block >>> (N_d, P_d, n);

// copy to cpu, output, free memory
   cudaMemcpy(P, P_d, size, cudaMemcpyDeviceToHost);
 
   /*for(int i=0; i<n; i++)
     for(int j=0; j<n; j++){
        cout<<i<<' '<<j<<' '<<P[i*n + j]<<endl;
   }*/

   cudaFree(N_d); cudaFree(P_d);
   free(N); free(P);

}
