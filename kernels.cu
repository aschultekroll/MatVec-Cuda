#include <cooperative_groups.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#define DTYPE float
namespace cg = cooperative_groups;




//First Kernel: Repeated call of a shared memory reduction kernel (Wiederholter Aufruf des Kernels, bis Ergebnis final)
__global__ void kernelRed(DTYPE *A, DTYPE *x, DTYPE *y, DTYPE *buf,int size, int iteration) {
  extern __shared__ DTYPE sdata[];
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  extern __shared__ DTYPE data[];
  int tx= threadIdx.y*blockDim.x + threadIdx.x;
  //Index Variable definieren
  int index = col + row * size;

  // matrox vektor multiplikation durchführen und ergebnis in buffer schrieben im ersten Aufruf des kernels
  if(iteration == 0){
    buf[index] = A[index]*x[col];
  }
  __syncthreads(); 


  if (col<size && row<size){
    //Ergebnis der multipilation in shared mamory schreiben
    sdata[tx] = buf[index];
    __syncthreads(); 

    //reduction
    for(unsigned int k = blockDim.x/2; k > 0; k >>= 1){ 
      if (threadIdx.x<k){
        sdata[tx] += sdata[tx + k];
        __syncthreads();

      }

    }



      

    //Ergebnis der Reduktion an den Anfang der Buffer Matrix schreiben und letzte werte auf null setzen
    buf[index] = (DTYPE)0;
    if(threadIdx.x == 0 ){
      buf[blockIdx.x+row*size] =sdata[tx];
      __syncthreads();
    }
    __syncthreads(); 


    //in der letzten Iteration wird der vorletzte wert der buffer matrix in den Ergebnis Vektor geschrieben
    if(iteration == floor(log2((DTYPE)size))-1){   
      if(threadIdx.x ==0 ){
        y[row]=buf[row*size];
        __syncthreads();
      }
    }
  }
}
//#######################################################################################################

//############################################################################################################


//Kernel 2.1: An Atomic operation at the end of the shared memory reduction kernel (Atomic Operations am Ende des Shared-Memory-Blocks)
__global__ void kernelRedAtomicAdd(DTYPE *A, DTYPE *x, DTYPE *y,int size) {
  extern __shared__ float sdata[];
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int index = col + row * size;
  int tx= threadIdx.x+ threadIdx.y*blockDim.x;

  if (col<size && row<size)
  {
    //Ergebnis der multipilation in shared mamory schreiben
    sdata[tx] = A[index] * x[col];
    __syncthreads();

    //reduktion
    for(int k = blockDim.x/2; k > 0; k >>= 1){
      if (threadIdx.x<k)
      sdata[tx] += sdata[tx + k];
      __syncthreads();
    }

    //atomic add zum berechnen der reduktion
    if(threadIdx.x ==0 ){
      atomicAdd(&y[row], sdata[threadIdx.y*blockDim.x]);
    }
  }
}
//##################################################################################


//Kernel 2.2: All calculations are performed by Atomic operations (Atomic Operations für alle notwendigen Operationen)
__global__ void kernelAtomic(DTYPE *A, DTYPE *x, DTYPE *y, int size) {

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int index = col + row * size;
  if (col<size && row<size){
    atomicAdd(&y[row], A[index] * x[col]);
  }
}
//################################################################################





//Third Kernel: Use of intra-grid_dim communication for synchronization of the shared memory kernel (Synchronisierung für SM-Reduktion über Threadblöcke (Kernel 1.2))
__device__ void reduceBlock(DTYPE *sdata, const cg::thread_block &block_dim)
{
    int bid = block_dim.thread_rank();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block_dim);

    double buf  = sdata[bid];

    //reduction
    for (int i = tile32.size() / 2; i > 0; i >>= 1) {
        if (tile32.thread_rank() < i) {
          sdata[bid] += sdata[bid+i];
        }
        tile32.sync();
    }
    block_dim.sync();

    //collect the reduction data, aggregate the data and write it to the 1st position of the shared memory
    if (bid == 0) {
        buf  = 0;
        for (int i = 0; i < blockDim.x; i += tile32.size()) {
            buf  += sdata[i];
        }
        sdata[0] = buf;
    }
    block_dim.sync();
}

__global__ void kernelIntraGrid(DTYPE *A, DTYPE *x, DTYPE*y, DTYPE* buf, int size) {
	extern __shared__ DTYPE sdata[];
	cg::thread_block block_dim = cg::this_thread_block();
  cg::grid_group grid_dim = cg::this_grid();
  int gid = grid_dim.thread_rank();
  int bid = block_dim.thread_rank();

    //Since not the whole grid_dim can be processed at once, this For loop is necessary
    for(int z = 0; z < size; z++){
      // Stride over grid_dim and add the values of the multiplication to a shared memory buffer
      sdata[bid] = 0;
      for (int i = gid; i < size; i += grid_dim.size()) {
          sdata[bid] += A[z * grid_dim.size() + i] * x[i];
      }

      block_dim.sync();

      // Reduce each block_dim (called once per block_dim)
      reduceBlock(sdata, block_dim);

      // Write out the result to global memory
      if (bid == 0) {
          buf[blockIdx.x] = sdata[0];
      }

      //interne synchronisierung der Blockweisen Reduktion 
      grid_dim.sync();

      //Calculate the result of the block_dim reduction and write it to the appropriate place in the result vector
      if (gid == 0) {
        DTYPE temp = (DTYPE)0;
        for (int t = 0; t < gridDim.x; t++) {
          temp += buf[t];
        }
        y[z] = temp;
      }
      grid_dim.sync();
    }
}
//############################################################################################################

//matrix-vektor produkt ohne shared memory
//GPU berechnet A*x=y
__global__ void kernelMatVec(DTYPE *A, DTYPE *x, DTYPE *y, int size)
{

   int i = threadIdx.x + blockIdx.x*blockDim.x;

      if (i<size) {
      DTYPE buf = 0.0;
      for (int j=0; j<size; j++) {     
         buf += A[j+i*size]*x[j];      
      }
      y[i] = buf;
   }
}