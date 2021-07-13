
#include <iostream>
#include <iomanip>
#include "kernels.cu"
#include "prep.cu"

#include <fstream>
using namespace std;

int main(int argc, char** argv) {
  // file pointer
  fstream fout;
  fout.open("data4.csv", ios::out | ios::app);
  std::ofstream myfile;
  myfile.open ("Messungen2.csv");

  // t
  int t = 1024;
  // i to calculate matrix size
  int i=16;




  //überprüfe, ob Eingabe gemacht wurde, falls ja verwende diese
  if (argc>1)
  {
    i=atoi(argv[1]);
  }
  int size = 1024 * i;

  //Kernel Konfiuration
  dim3 block_dim(32,32);
  dim3 grid_dim(size/block_dim.x, size/block_dim.y);


  // Datenfelder anlegen
  DTYPE *y_dev, *b_dev, *a_dev, *x_dev;
  DTYPE *yd_host, *yh_host, *a_host, *x_host;
  

  // allocate pointers for host
  x_host = (DTYPE *)malloc(size * sizeof(DTYPE));
  yh_host = (DTYPE *)malloc(size * sizeof(DTYPE));
  a_host = (DTYPE *)malloc(size * size * sizeof(DTYPE));
  yd_host = (DTYPE *)malloc(size * sizeof(DTYPE));

  // Matrix und Vektor füllen
  fillVector(x_host, size);
  fillMatrix(a_host, size);

  //cuda Memory
  cudaMalloc((void **)&y_dev, size * sizeof(DTYPE));
  cudaMalloc((void **)&b_dev, size * size * sizeof(DTYPE));
  cudaMalloc((void **)&a_dev, size * size * sizeof(DTYPE));
  cudaMalloc((void **)&x_dev, size * sizeof(DTYPE));
 

  cudaEvent_t start,end;
  struct timespec start_h,end_h;

  // Zeiten auf numm setzen
  float hostToDevTime=0.0;
  float devToHostTime= 0.0;
  float hostTime = 0.0;
  float atomicKernelTime = 0.0;
  float sharedMemKernelTime = 0.0;
  float sharedMemAtomicAddKernelTime = 0.0;
  float kernelIntraGridTime =0.0;
  
  // events for time measurement
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  //start time measurement
  cudaEventRecord(start);

  // Daten von Host zu device kopieren
  cudaMemcpy(a_dev, a_host, size * size * sizeof(DTYPE),cudaMemcpyHostToDevice);
  cudaMemcpy(x_dev, x_host, size * sizeof(DTYPE), cudaMemcpyHostToDevice);

  //Zeitmessung beenden
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&hostToDevTime, start, end);

  printf("Zeit um Daten von Host zu Device zu kopieren (ms): %f\n", hostToDevTime);
  printf("In Sekunden: %f\n",hostToDevTime/1000);
  printf("\n");

  //check for errorors
  cudaError_t error=cudaGetLastError();
  if (error!=cudaSuccess)
  {
    printf("Fehler im Speichermanagement!!!: %s (%i)\n",cudaGetErrorString(error),error); //TODO
    return(-1);
  }
  


  //############################ CPU Zeitmessung ############################
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&start_h);
  hostAx(a_host, x_host, yh_host, size);
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&end_h);
  hostTime=(double)((end_h.tv_nsec+end_h.tv_sec*1E9) - (start_h.tv_nsec+start_h.tv_sec*1E9))/1E6;
  printf("Zeit auf CPU in Sekunden: %f\n", hostTime/1000);
  printf("\n");
  //####################################################################################



  //######################## Shared Memory Kernel (Kernel 1) ###############################
  //set chache configuration
  //cudaFuncSetCacheConfig(kernelRed, cudaFuncCachePreferEqual);
  //cudaFuncSetCacheConfig(kernelRed, cudaFuncCachePreferShared);
  //cudaFuncSetCacheConfig(kernelRed, cudaFuncCachePreferL1);

  cudaMemset(y_dev,0,size*sizeof(DTYPE));
  int sdata_sz = (block_dim.x*block_dim.y)*sizeof(DTYPE);
  cudaEventRecord(start);
  // memory kernels ln(size) mal aufrufen
  for(int i = 0; i < floor(log2((DTYPE)size)); i++){
    kernelRed<<<grid_dim, block_dim, sdata_sz>>>(a_dev, x_dev, y_dev, b_dev, size, i); 
  }
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&sharedMemKernelTime, start, end);
  cudaDeviceSynchronize();

  error=cudaGetLastError();
  if (error!=cudaSuccess)
  {
    printf("Fehler im Shared Memory Kernel!!!: %s (%i)\n",cudaGetErrorString(error),error);
    return(-1);
  }

  cudaMemcpy(yd_host, y_dev, size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
  correctness(a_host, yd_host, yh_host, x_host, size);
  printf("Time Kernel 1 (seconds): %f\n", sharedMemKernelTime/1000);
  printf("\n");
  //########################################################################################


  //#################### Kernel 2.1 - shared memory with atomic operations at the end of a block_dim ########################
    //cudaFuncSetCacheConfig(kernelRedAtomicAdd, cudaFuncCachePreferEqual);
  //cudaFuncSetCacheConfig(kernelRedAtomicAdd, cudaFuncCachePreferShared);
  //cudaFuncSetCacheConfig(kernelRedAtomicAdd, cudaFuncCachePreferL1);
  cudaMemset(y_dev,0,size*sizeof(float));
  cudaEventRecord(start);
  kernelRedAtomicAdd<<<grid_dim, block_dim,sdata_sz>>>(a_dev, x_dev, y_dev, size);
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&sharedMemAtomicAddKernelTime
, start, end);
  cudaDeviceSynchronize();

  error=cudaGetLastError();
  if (error!=cudaSuccess)
  {
    printf("Error in Kernel 2.1 !!!: %s (%i)\n",cudaGetErrorString(error),error);
    return(-1);
  }
  cudaEventRecord(start);
  cudaMemcpy(yd_host, y_dev, size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&devToHostTime, start, end);

  //Ergebnis Überprüfen
  correctness(a_host, yd_host, yh_host, x_host, size);
  //Zeit ausgeben
  printf("Zeit Kernel 2.1 (ms): %f\n", sharedMemAtomicAddKernelTime
);
  printf("in Sekunden:%f\n",sharedMemAtomicAddKernelTime
/1000);
  printf("\n");
  //########################################################################################

  // ################## Kernel 2.2 - all operations atomic ######################
    //cudaFuncSetCacheConfig(kernelAtomic, cudaFuncCachePreferEqual);
  //cudaFuncSetCacheConfig(kernelAtomic, cudaFuncCachePreferShared);
  //cudaFuncSetCacheConfig(kernelAtomic, cudaFuncCachePreferL1);
  cudaMemset(y_dev,0,size*sizeof(float));
  cudaEventRecord(start);
  kernelAtomic<<<grid_dim, block_dim>>>(a_dev, x_dev, y_dev, size);
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&atomicKernelTime, start, end);
  cudaDeviceSynchronize();

  error=cudaGetLastError();
  if (error!=cudaSuccess)
  {
    printf("Fehler in Kernel 2.2 !!! : %s (%i)\n",cudaGetErrorString(error),error);
    return(-1);
  }

  cudaMemcpy(yd_host, y_dev, size * sizeof(DTYPE), cudaMemcpyDeviceToHost);

  //Ergebnis Überprüfen
  correctness(a_host, yd_host, yh_host, x_host, size);

  //Zeit ausgeben
  printf("Zeit Kernel 2.2 (ms): %f\n", atomicKernelTime);
  printf("in Sekunden: %f\n", atomicKernelTime/1000);
  printf("\n");
  //####################################################################################-


  //################## Third Kernel: Intra Grid Groups (Kernel 3) #####################
//cudaFuncSetCacheConfig(kernelIntraGrid, cudaFuncCachePreferEqual);
//cudaFuncSetCacheConfig(kernelIntraGrid, cudaFuncCachePreferShared);
  //cudaFuncSetCacheConfig(kernelIntraGrid, cudaFuncCachePreferL1);
  


  block_dim.x = t;
  block_dim.y = 1;

  DTYPE* buf;
  cudaMalloc((void **)&buf, size * sizeof(DTYPE));
  cudaMemset(y_dev,0,size*sizeof(float));
  cudaMemset(buf,0,size*sizeof(float));

  //Berechnen der device occupancy, um zu wissen, wie viele Blöcke gleichzeitig ausgeführt werden können
  int numBlocksPerSm;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernelIntraGrid, t, t*sizeof(DTYPE));

  int num_sums = deviceProp.multiProcessorCount;
  int n_blocks = min(numBlocksPerSm*num_sums, (size+t-1)/t);

  grid_dim.x = n_blocks;
  grid_dim.y = 1;
 
  void *params[] = {(void *)&a_dev, (void *)&x_dev, (void *)&y_dev,(void*)&buf, (void *)&size};
  sdata_sz = t*sizeof(DTYPE);
  cudaEventRecord(start);
  
  cudaLaunchCooperativeKernel((void *)kernelIntraGrid, grid_dim, block_dim, params, sdata_sz, NULL);
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&kernelIntraGridTime, start, end);
 
  cudaMemcpy(yd_host, y_dev, size * sizeof(DTYPE), cudaMemcpyDeviceToHost);

  
  error=cudaGetLastError();
  if (error!=cudaSuccess)
  {
    printf("Fehler im intra Grid Kernel!!! : %s (%i)\n",cudaGetErrorString(error),error);
    return(-1);
  }

  cudaMemcpy(yd_host, y_dev, size * sizeof(DTYPE), cudaMemcpyDeviceToHost);

  //Ergebnis Überprüfen
  correctness(a_host, yd_host, yh_host, x_host, size);

  //Zeit ausgeben
  printf("Zeit Intra Grid Kernel (ms): %f\n", kernelIntraGridTime);
  printf("in Sekunden: %f\n", kernelIntraGridTime/1000);
  printf("\n");
  //#############################################################################################################

  // Zeiten in Datei schreiben
  string cache_config ="PrefereNone";
  string gpu = "GPU01";
  //fout << "ArraySize; tX; tY; GridSizeX; GridSizeY; Kernel_1; Kernel_2.1; Kernel_2.2; Kernel_1.2; HostTime; HostToDevice; DeviceToHost; Chache_Configuration;GPU \n";
  //fout <<  size << ";" << block_dim.x << ";" <<block_dim.y<< ";"<<grid_dim.x<< ";"<< grid_dim.y<< ";"<< sharedMemKernelTime << ";"<<sharedMemAtomicAddKernelTime
 //<< ";"<< atomicKernelTime << ";"<< kernelIntraGridTime<< ";"<< hostTime<< ";"<< hostToDevTime<< ";"<< devToHostTime <<";" << cache_config << ";"<<gpu <<"\n";

  
  myfile.close();

  //destroy cuda events
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  //free memory for device and host
  cudaFree(a_dev);
  cudaFree(y_dev);
  cudaFree(x_dev);
  cudaFree(b_dev);

  free(a_host);
  free(yd_host);
  free(yh_host);
  free(x_host);
  return 0;
}
