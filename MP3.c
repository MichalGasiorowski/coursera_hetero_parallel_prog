// MP 3: Due Sunday, Dec 30, 2012 at 11:59 p.m. PST
#include    <wb.h>

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

#define TILE_WIDTH 8
      
// Compute C = A * B
__global__ void matrixMultiplyShared(float * A, float * B, float * C,
			             int numARows, int numAColumns,
			             int numBRows, int numBColumns,
			             int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
    //@@ You have to use shared memory for this MP
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
  

	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x;int ty = threadIdx.y;
	
  	int Row = by * TILE_WIDTH + ty;
  	int Col = bx * TILE_WIDTH + tx;
  	//if(Row>=numAColumns || Col>=numBColumns) return;
  	
  	float Pvalue = 0;
  	int maxCols = max(numAColumns, numBColumns);
  	int maxRows = max(numARows, numBRows);	
  	
  	for (int m = 0; m < (numAColumns - 1)/TILE_WIDTH + 1; ++m) {
    	if((Row < numARows) && ((m*TILE_WIDTH + tx) < numAColumns)) {
      		ds_A[ty][tx] = A[Row*numAColumns + m * TILE_WIDTH + tx];
    	} else {
      		ds_A[ty][tx] = 0;
    	}
      	if((m * TILE_WIDTH + ty < numBRows) && (Col < numBColumns)) {
        	ds_B[ty][tx] = B[(m*TILE_WIDTH + ty)*numBColumns + Col];
      	} else {
        	ds_B[ty][tx] = 0;
      	}
    	__syncthreads();
      	
    	if((Row < numCRows) && (Col < numCColumns)) {
      		for (int k = 0; k < TILE_WIDTH; ++k) {
        		Pvalue += ds_A[ty][k] * ds_B[k][tx];
      		}
    	}
    	__syncthreads(); 
    }
  	
  
  	//if(Row == 0 && Col == 0)
    //    printf("(%d, %d): %f \n", Row, Col, Pvalue);
  	if ((Row < numCRows) && (Col < numCColumns)) {
      	C[Row*numCColumns + Col] = Pvalue;
      	//if(Col == 0 && Row == 56)
    	//	printf("(%d, %d): %f \n", Row, Col, Pvalue);

  	}
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * deviceA;
    float * deviceB;
    float * deviceC;
    int numARows; // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows; // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
    //@@ Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;
  
  	int numCSize = numCRows * numCColumns * sizeof(float);
    //@@ Allocate the hostC matrix
    hostC = (float*) malloc(numCRows * numCColumns * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    cudaMalloc((void**)&deviceA, numARows * numAColumns * sizeof(float));
    cudaMalloc((void**)&deviceB, numBRows * numBColumns * sizeof(float));
	cudaMalloc((void**)&deviceC, numCSize);

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);

    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here
    // int TILE_SIZE = 16;
  	int max_rows = max(numARows, numBRows);
  	int max_cols = max(numAColumns, numBColumns);
  	int max_size = max(max_rows, max_cols);
      
    //int rowSize = ceil(max_size/(float)TILE_WIDTH);
  	//int colSize = ceil(max_size/(float)TILE_WIDTH);
  
  	int rX = (numCRows - 1)/TILE_WIDTH + 1;
  	int rY = (numCColumns - 1)/TILE_WIDTH + 1;
  	
  	dim3 dimGrid(rX, rY, 1);
  	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
     
    
  	wbLog(TRACE, "(ARows, ACols): (",numARows, ", " , numAColumns, ")");
  	wbLog(TRACE, "(BRows, BCols): (",numBRows, ", " , numBColumns, ")");
  	wbLog(TRACE, "(CRows, CCols): (",numCRows, ", " , numCColumns, ")");
  	wbLog(TRACE, "dimGrid is (",rX, ", " , rY, ") , dimBlock is ", TILE_WIDTH);
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
	matrixMultiplyShared<<<dimGrid, dimBlock>>> (deviceA, deviceB, deviceC,
			       numARows, numAColumns,
			       numBRows, numBColumns,
			       numCRows, numCColumns);  
    cudaThreadSynchronize();
    
    wbTime_stop(Compute, "Performing CUDA computation");
    
    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostC, deviceC, numCSize, cudaMemcpyDeviceToHost);

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
	cudaFree(deviceA);
	cudaFree(deviceB);
  	cudaFree(deviceC);
  
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}

