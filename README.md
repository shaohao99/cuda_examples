This directory shows some CUDA examples.

1. mat_mul: Matrix multiplication examples.
mat_mul.cu: Compute matrix multiplication using global memory.
mat_mul_tile.cu: Compute matrix multiplication using shared memory. The matrix is divided into small tiles and each tile is saved in shared memory, in order to gain large bandwidth. 

2. prefix_sum: Compute prefix sum of an array. Use shared memory. The program Works for arbitrary array size (e.g. the size can be larger than a block size). 

3. conv2d: Compute 2D convolution. The filter is saved in read-only constant memory to gain large bandwidth. 

4. vec_add_multi_gpus: Compute vector addition using multiple GPUs on onde node of a computer cluster. There is no data communication between GPUs in vector addition. The number of GPUs on the node is automatically detected and all GPUs on the node are used. OpenMP is used for CPU multithreading. Each CPU thread assigns a CUDA kernel to one GPU. 

5. conv_cudnn: Build a typical convolution layer in CNN. The CUDNN library is used to compute the convolution and activation processes. The OpenCV library is used to load, normalize and save images.  

