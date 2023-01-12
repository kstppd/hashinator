CC=nvcc  
LNK=nvcc 
OPT=  -Xcompiler -O3 
EXTRA=  --std=c++17
EXTRA+= -gencode arch=compute_60,code=sm_60  -L /usr/lib/x86_64-linux-gnu
OBJ= hashmap.o 

default: examples
examples: ${OBJ}

allclean:
	rm benchmark

hashmap.o: hashmap.cu
	${CC} ${OPT} ${EXTRA}   -o benchmark hashmap.cu
