CC=nvcc  
LNK=nvcc 
OPT=  -Xptxas -O3 
CXXFLAGS=
EXTRA=  --std=c++14
EXTRA+= -gencode arch=compute_60,code=sm_60  -L /usr/lib/x86_64-linux-gnu
EXTRA+=  -DHASHMAPDEBUG --expt-relaxed-constexpr 
GTEST=-lgtest -lgtest_main -lpthread
OBJ= gtest_vec_host.o	gtest_vec_device.o  map_add_remove.o stream_compaction.o delete_mechanism.o

default: all
all: ${OBJ}

clean: 
	rm  ${OBJ}
allclean:
	rm ${OBJ} &
	rm gtestvechost &
	rm gtestvecdevice &
	rm map_add_remove & 
	rm compaction &
	rm delete_mechanism

map_add_remove.o: add_remove_map_elements/main.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST}  -o map_add_remove add_remove_map_elements/main.cu

gtest_vec_host.o: gtest_vec_host/vec_test.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST} -o gtestvechost gtest_vec_host/vec_test.cu

gtest_vec_device.o: gtest_vec_device/vec_test.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST} -o gtestvecdevice gtest_vec_device/vec_test.cu

stream_compaction.o: stream_compaction/race.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST} -o compaction stream_compaction/race.cu

delete_mechanism.o: delete_by_compaction/main.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST} -o delete_mechanism delete_by_compaction/main.cu

