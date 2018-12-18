COMPILER	:= g++

CUDA_INSTALL_PATH ?= (fill this in)

NVCC = $(CUDA_INSTALL_PATH)/(fill this in)
INCD = -l$(CUDA_INSTALL_PATH)/(fill this in)
LIBS = -L$(CUDA_INSTALL_PATH)/(fill this in) -lcudart -lcublas -lcusolver -fopenmp 

main: kernels.o SAP.o example_call.o
	$(COMPILER) $(LIBS) kernels.o SAP.o example_call.o
example_call.o: example_call.cu
	$(NVCC) -c $(INCD) example_call.cu
SAP.o: SAP.cu
	$(NVCC) -c $(INCD) SAP.cu
kernels.o: kernels.cu
	$(NVCC) -c $(INCD) kernels.cu

clean: 
	rm  *.o
