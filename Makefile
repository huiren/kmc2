NVCCFLAG = -arch=sm_30 -dlink -dc -O3 -lnvToolsExt -Xcompiler -fopenmp -Xlinker -lgomp 
all: app

app: kmc2.gpu.o mmer.o
	nvcc kmc2.gpu.o mmer.o -lnvToolsExt -O3 -Xcompiler -fopenmp -Xlinker -lgomp -o app

kmc2.gpu.o: kmc2.gpu.cu
	nvcc -c $(NVCCFLAG) kmc2.gpu.cu
mmer.o: mmer.cu mmer.h
	nvcc -c $(NVCCFLAG) mmer.cu

clean:
	rm *.o app
	rm *.bin
