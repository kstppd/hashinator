CC=nvcc  
LNK=nvcc 
OPT=  -Xptxas -O3 
CXXFLAGS=
EXTRA= -g  --std=c++14
EXTRA+= -gencode arch=compute_60,code=sm_60  -L /usr/lib/x86_64-linux-gnu
EXTRA+=  -DHASHMAPDEBUG --expt-relaxed-constexpr 
GTEST=-lgtest -lgtest_main -lpthread
OBJ= gtest_vec_host.o	gtest_vec_device.o  gtest_hashmap.o stream_compaction.o delete_mechanism.o

default: all
all: ${OBJ}

clean: 
	rm  ${OBJ}
allclean:
	rm ${OBJ} &
	rm gtestvechost &
	rm gtestvecdevice &
	rm gtest_hashmap & 
	rm compaction &
	rm delete_mechanism

gtest_hashmap.o: hashmap_unit_test/main.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST}  -o gtest_hashmap hashmap_unit_test/main.cu

gtest_vec_host.o: gtest_vec_host/vec_test.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST} -o gtestvechost gtest_vec_host/vec_test.cu

gtest_vec_device.o: gtest_vec_device/vec_test.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST} -o gtestvecdevice gtest_vec_device/vec_test.cu

stream_compaction.o: stream_compaction/race.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST} -o compaction stream_compaction/race.cu

delete_mechanism.o: delete_by_compaction/main.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST} -o delete_mechanism delete_by_compaction/main.cu

