// MP 5 Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}
// Due Tuesday, January 22, 2013 at 11:59 p.m. PST
 
#include    <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)
      
__global__ void add(float * output, float * sums, int len) {
 	unsigned int t = threadIdx.x;	
  	unsigned int bi = blockIdx.x;
  	unsigned int i = bi*blockDim.x + threadIdx.x;
  	
  	if(i >= len)
      	return;
  
  	for (unsigned int k = 0;k<bi;k++) {
    	output[i] += sums[k];
      	__syncthreads();
  	}
  	/*
  	if(bi > 0) {
  		output[i] += sums[bi-1];
      	if(t==0) {
      		printf("\nbi: %d",bi);
          	printf("\n output[i]: %f",output[i]);
      		printf("\n sums: %f",sums[bi-1]);
      	}
  	}
    */
    
}
__global__ void scan(float * input, float * output, float* sums, int len) {
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here
      
    __shared__ float scan_array[2*BLOCK_SIZE];	
  	unsigned int t = threadIdx.x; 
  	unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;

	if(i < len) 
      	scan_array[t] = input[i]; 
  	else scan_array[t] = 0;
  	
  	if((i+blockDim.x) < len) 
    	scan_array[t+blockDim.x] = input[i+blockDim.x]; 
  	else scan_array[t+blockDim.x] = 0;
  	
   	int stride = 1;
  	while(stride < BLOCK_SIZE)
    {
     int index = (t + 1)*stride*2 -1;
     if(index < BLOCK_SIZE)
     	scan_array[index] += scan_array[index - stride];
     	stride = stride *2;
     __syncthreads();
    }
  	
  	for(int stride = BLOCK_SIZE/2;stride > 0; stride/=2) {
  		__syncthreads();
    	int index = (t + 1)*stride*2 - 1;
    	if(index + stride < BLOCK_SIZE) {
   			scan_array[index + stride] += scan_array[index];
    	}
  	}
  	__syncthreads();
  	if(i < len)
  		output[i] = scan_array[t];
  	if(t + 1 == blockDim.x) {
    	//printf("\nLast one: %f",output[i]) ;
      	sums[blockIdx.x] = output[i];
  	}
}


  	
  	
  	
  	


int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
  	
  	float * deviceSums;
  
    int numElements; // number of elements in the list
	int numOfBlocks;
  	
  
    args = wbArg_read(argc, argv);
  
	
    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
  
  	numOfBlocks = (numElements - 1)/BLOCK_SIZE + 1;
  	
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);
	wbLog(TRACE, "The number of blocks is ", numOfBlocks);
    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
  	wbCheck(cudaMalloc((void**)&deviceSums, numOfBlocks*sizeof(float)));
  	
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");
	
    //@@ Initialize the grid and block dimensions here
    
	
  	
  
	dim3 dimGrid(numOfBlocks, 1, 1);
  	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	
  	
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
      
      
	scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, deviceSums, numElements);
  	cudaDeviceSynchronize();
  	add<<<dimGrid, dimBlock>>>(deviceOutput, deviceSums, numElements);
    cudaDeviceSynchronize();
  	
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
  	cudaFree(deviceSums);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}

