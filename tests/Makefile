CC=nvcc  
LNK=nvcc 
OPT=
CXXFLAGS= -g -G 
EXTRA= -Xcompiler -rdynamic --std=c++11
EXTRA+= -gencode arch=compute_60,code=sm_60 -rdc=true -L /usr/lib/x86_64-linux-gnu
EXTRA+=-DCUDAVEC

OBJ= gpuport.o speed_test.o vec_test.o mempool_main.o vec_of_vec_test.o


all: ${OBJ}

clean: 
	rm  *.o 
allclean:
	rm gpuport &
	rm speed_test &
	rm vec_test &
	rm vec_of_vec &
	rm mempool


gpuport.o: hash_gpu_vs_cpu/gpuport.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA}  -o gpuport  hash_gpu_vs_cpu/gpuport.cu 

speed_test.o: speed_tests/speed_test.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA}  -o speed_test speed_tests/speed_test.cu 

vec_test.o: vector_tests/vec_test.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA}  -o vec_test vector_tests/vec_test.cu 

vec_of_vec_test.o: vector_of_vectors/vec_of_vec_test.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA}  -o vec_of_vec vector_of_vectors/vec_of_vec_test.cu
mempool_main.o: mempool_test/mempool_main.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA}  -o mempool mempool_test/mempool_main.cu 
