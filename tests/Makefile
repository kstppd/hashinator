CC=nvcc  
LNK=nvcc 
OPT= -O3 
CXXFLAGS= -g -G 
EXTRA= -Xcompiler -rdynamic --std=c++14
EXTRA+= -gencode arch=compute_60,code=sm_60 -rdc=true -L /usr/lib/x86_64-linux-gnu
EXTRA+=  -DHASHMAPDEBUG --expt-relaxed-constexpr

GTEST=-lgtest -lgtest_main -lpthread
OBJ= map_gpu_vs_cpu.o speed_test.o	  mempool_main.o	  gtest_vec.o	gtest_vec_host.o	  gtest_hashmap.o	  map_add_remove.o 	  vecspeedtest

default: all
all: ${OBJ}

clean: 
	rm  *.o 
allclean:
	rm map_gpu_vs_cpu &
	rm speed_test &
	rm vec_test &
	rm vec_of_vec &
	rm gtestvec &
	rm gtestvechost &
	rm gtesthashmap &
	rm mempool &
	rm map_rw &
	rm map_add_remove & 
	rm vecspeedtest


map_gpu_vs_cpu.o: hash_gpu_vs_cpu/gpuport.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA}  -o map_gpu_vs_cpu  hash_gpu_vs_cpu/gpuport.cu 

speed_test.o: speed_tests/speed_test.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA}  -o speed_test speed_tests/speed_test.cu 

gtest_hashmap_rw.o: gtest_hashmap_rw/main.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST}  -o map_rw gtest_hashmap_rw/main.cu

map_add_remove.o: add_remove_map_elements/main.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST}  -o map_add_remove add_remove_map_elements/main.cu

mempool_main.o: mempool_test/mempool_main.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA}  -o mempool mempool_test/mempool_main.cu 

gtest_vec.o: gtest_vec/vec_test.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST} -o gtestvec gtest_vec/vec_test.cu

gtest_vec_host.o: gtest_vec_host/vec_test.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST} -o gtestvechost gtest_vec_host/vec_test.cu

vecspeedtest: vec_speed_test/main.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} -o vecspeedtest vec_speed_test/main.cu -DPROFINY_CALL_GRAPH_PROFILER -lboost_timer -lboost_chrono -lboost_system

gtest_hashmap.o: gtest_hasmap/unit.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST} -o gtesthashmap gtest_hasmap/unit.cu

