#include <cudnn.h>
#include <opencv2/opencv.hpp>

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }


cv::Mat load_image(const char* image_path) {   // use opencv to load and normalize an image
    cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
    image.convertTo(image, CV_32FC3);
    cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
    return image;
}

// a function to save output image using opencv
void save_image(const char* output_filename,
                float* buffer,
                int height,
                int width) {
  cv::Mat output_image(height, width, CV_32FC3, buffer);
  // Make negative values zero.
  cv::threshold(output_image,
                output_image,
                /*threshold=*/0,
                /*maxval=*/0,
                cv::THRESH_TOZERO);
  cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
  output_image.convertTo(output_image, CV_8UC3);
  cv::imwrite(output_filename, output_image);
}



int main(int argc, char const *argv[]) {

  cudnnHandle_t my_cudnn;
  //cudnnCreate(&my_cudnn);
  checkCUDNN(cudnnCreate(&my_cudnn));

  // use opencv to load and normalize an image
  cv::Mat image = load_image("/projectnb/scv/shaohao/data/cifar/train/1898_horse.png");

  int batch_size = 1;
  int inp_channels =3;
  int out_channels =3;
  int height = image.rows; 
  int width = image.cols;

  cudnnTensorDescriptor_t input_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,    // define a 4d tensor to hold an input image
                                      /*format=*/CUDNN_TENSOR_NHWC,  // layout order of the following arguments
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/1,
                                      /*channels=*/3,
                                      /*image_height=*/image.rows,
                                      /*image_width=*/image.cols));

  cudnnTensorDescriptor_t output_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,    // define a 4d tensor to hold an input image
                                      /*format=*/CUDNN_TENSOR_NHWC,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/1,
                                      /*channels=*/3,
                                      /*image_height=*/image.rows,
                                      /*image_width=*/image.cols));

  cudnnFilterDescriptor_t kernel_descriptor;    // define a 4d tensor to hold a filter
  checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
  checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*format=*/CUDNN_TENSOR_NCHW,
                                      /*out_channels=*/3,
                                      /*in_channels=*/3,
                                      /*kernel_height=*/3,
                                      /*kernel_width=*/3));


   cudnnConvolutionDescriptor_t convolution_descriptor;  // define convolution parameters
   checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
   checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                           /*pad_height=*/1,
                                           /*pad_width=*/1,
                                           /*vertical_stride=*/1,
                                           /*horizontal_stride=*/1,
                                           /*dilation_height=*/1,
                                           /*dilation_width=*/1,
                                           /*mode=*/CUDNN_CROSS_CORRELATION,
                                           /*computeType=*/CUDNN_DATA_FLOAT));


  cudnnConvolutionFwdAlgo_t convolution_algorithm;  // define convolutoin algorithm
  checkCUDNN( cudnnGetConvolutionForwardAlgorithm(my_cudnn,
                                        input_descriptor,
                                        kernel_descriptor,
                                        convolution_descriptor,
                                        output_descriptor,
                                        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                        /*memoryLimitInBytes=*/0,
                                        &convolution_algorithm));

  size_t workspace_bytes = 0;   // estimate work space size
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(my_cudnn,
                                                   input_descriptor,
                                                   kernel_descriptor,
                                                   convolution_descriptor,
                                                   output_descriptor,
                                                   convolution_algorithm,
                                                   &workspace_bytes));
  std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB" << std::endl;

// allocate memory
  void* d_workspace{nullptr};
  cudaMalloc(&d_workspace, workspace_bytes);

  int inp_image_bytes = batch_size * inp_channels * height * width * sizeof(float);
  int out_image_bytes = batch_size * out_channels * height * width * sizeof(float);

  float* d_input{nullptr};
  cudaMalloc(&d_input, inp_image_bytes);
  cudaMemcpy(d_input, image.ptr<float>(0), inp_image_bytes, cudaMemcpyHostToDevice);  // copy input image from host to GPU

  float* d_output{nullptr};
  cudaMalloc(&d_output, out_image_bytes);
  cudaMemset(d_output, 0, out_image_bytes);  // set GPU memory for output image

// Initialize filter
  const float kernel_template[3][3] = {
    {1,  1, 1},
    {1, -8, 1},
    {1,  1, 1}
  };

  float h_kernel[3][3][3][3];
  for (int kernel = 0; kernel < 3; ++kernel) {
    for (int channel = 0; channel < 3; ++channel) {
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
          h_kernel[kernel][channel][row][column] = kernel_template[row][column];
        }
      }
    }
  }

  float* d_kernel{nullptr};
  cudaMalloc(&d_kernel, sizeof(h_kernel));
  cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);  // copy filter from host to GPU

// Compute convolution on GPU
  const float alpha = 1, beta = 0;  // dstValue = alpha[0]*result + beta[0]*priorDstValue
  checkCUDNN(cudnnConvolutionForward(my_cudnn, 
                                   &alpha,
                                   input_descriptor,
                                   d_input,
                                   kernel_descriptor,
                                   d_kernel,
                                   convolution_descriptor,
                                   convolution_algorithm,
                                   d_workspace,
                                   workspace_bytes,
                                   &beta,
                                   output_descriptor,
                                   d_output));

// copy output image from GPU memory to CPU memory
  float* h_output = new float[out_image_bytes];
  cudaMemcpy(h_output, d_output, out_image_bytes, cudaMemcpyDeviceToHost);
  save_image("convolution-out.png", h_output, height, width);  // save output image

// Describe the activation
  cudnnActivationDescriptor_t activation_descriptor;
  checkCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
  checkCUDNN(cudnnSetActivationDescriptor(activation_descriptor,
                                        /*mode=*/CUDNN_ACTIVATION_SIGMOID,
                                        /*reluNanOpt=*/CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/0));

// Perform the forward pass of the activation
  checkCUDNN(cudnnActivationForward(my_cudnn,
                                  activation_descriptor,
                                  &alpha,
                                  output_descriptor,
                                  d_output,
                                  &beta,
                                  output_descriptor,
                                  d_output));

// copy output image from GPU memory to CPU memory
  cudaMemcpy(h_output, d_output, out_image_bytes, cudaMemcpyDeviceToHost);
  save_image("activation-out.png", h_output, height, width);  // save output image

// Release resources
  delete[] h_output;
  cudaFree(d_kernel);
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_workspace);
  cudnnDestroyTensorDescriptor(input_descriptor);
  cudnnDestroyTensorDescriptor(output_descriptor);
  cudnnDestroyFilterDescriptor(kernel_descriptor);
  cudnnDestroyConvolutionDescriptor(convolution_descriptor);
  cudnnDestroyActivationDescriptor(activation_descriptor);
  cudnnDestroy(my_cudnn);

}  // end of main
