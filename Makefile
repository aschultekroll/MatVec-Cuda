run: MatrixVectorCuda.cu
	nvcc  -o matvec MatrixVectorCuda.cu -arch=sm_60 -rdc=true --expt-relaxed-constexpr 


debug: MatrixVectorCuda.cu
		nvcc -g -G -arch=sm_60 MatrixVectorCuda.cu -o matvec -rdc=true -Xptxas -v --expt-relaxed-constexpr 


