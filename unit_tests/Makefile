CC=nvcc  
LNK=nvcc 
OPT= -O3 -Xcompiler 
CXXFLAGS= 
EXTRA= -g3  --std=c++17
EXTRA+= -gencode arch=compute_60,code=sm_60  -L /usr/lib/x86_64-linux-gnu
EXTRA+=  -DHASHMAPDEBUG --expt-relaxed-constexpr  --expt-extended-lambda
GTEST=-lgtest -lgtest_main -lpthread
OBJ= gtest_vec_host.o	gtest_vec_device.o  gtest_hashmap.o stream_compaction.o delete_mechanism.o insertion_mechanism.o hybrid_cpu.o hybrid_gpu.o pointer_test.o benchmark.o benchmarkLF.o tbPerf.o realistic.o stream_compaction2


default: tests
tests: ${OBJ}

clean: 
	rm  ${OBJ}
allclean:
	rm ${OBJ} &
	rm gtestvechost &
	rm gtestvecdevice &
	rm gtest_hashmap & 
	rm compaction &
	rm compaction2 &
	rm delete_mechanism &
	rm hybrid_cpu & 
	rm hybrid_gpu &
	rm pointertest &
	rm benchmark_hashinator &
	rm benchmark_hashinator_lf &
	rm benchmark_hashinator_tb &
	rm benchmark_hashinator_rl &
	rm insertion

gtest_hashmap.o: hashmap_unit_test/main.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST}  -o gtest_hashmap hashmap_unit_test/main.cu

benchmark.o: benchmark/main.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST}  -o benchmark_hashinator benchmark/main.cu

tbPerf.o: benchmark/tbPerf.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST}  -o benchmark_hashinator_tb benchmark/tbPerf.cu

realistic.o: benchmark/realistic.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST}  -o benchmark_hashinator_rl benchmark/realistic.cu

benchmarkLF.o: benchmark/loadFactor.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST}  -o benchmark_hashinator_lf benchmark/loadFactor.cu

gtest_vec_host.o: gtest_vec_host/vec_test.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST} -o gtestvechost gtest_vec_host/vec_test.cu

gtest_vec_device.o: gtest_vec_device/vec_test.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST} -o gtestvecdevice gtest_vec_device/vec_test.cu

stream_compaction.o: stream_compaction/race.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST} -o compaction stream_compaction/race.cu

stream_compaction2.o: stream_compaction/unit.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST} -o compaction2 stream_compaction/unit.cu

delete_mechanism.o: delete_by_compaction/main.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST} -o delete_mechanism delete_by_compaction/main.cu

insertion_mechanism.o: insertion_mechanism/main.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST} -o insertion insertion_mechanism/main.cu

pointer_test.o: pointer_test/main.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST} -o pointertest pointer_test/main.cu

hybrid_gpu.o: hybrid/main.cpp
	${CC} -x cu  ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST} -o hybrid_gpu hybrid/main.cpp

hybrid_cpu.o: hybrid/main.cpp
	g++ -fsanitize=address -fno-omit-frame-pointer ${CXXFLAGS} -O3  -std=c++17 -DHASHINATOR_CPU_ONLY_MODE  -o hybrid_cpu hybrid/main.cpp -I/usr/local/cuda/include/ -L/usr/local/cuda/lib64 -lcudart -lcuda  -lgtest -lgtest_main -lpthread  -Wall  -Wpedantic -Werror -static-libasan
