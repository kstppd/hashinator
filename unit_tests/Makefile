CC=nvcc  
CCC=clang++
LNK=nvcc 
OPT= -O3 -Xcompiler -fopenmp
CXXFLAGS= 
EXTRA=  --std=c++17
EXTRA+= -gencode arch=compute_60,code=sm_60  
EXTRA+=  -DHASHMAPDEBUG --expt-relaxed-constexpr  --expt-extended-lambda -lpthread
GTEST= -L/home/kstppd/libs/googletest/build/lib  -I/home/kstppd/libs/googletest/googletest/include -lgtest -lgtest_main -lpthread
OBJ= gtest_vec_host.o	gtest_vec_device.o gtest_device_vec.o  gtest_hashmap.o stream_compaction.o stream_compaction2.o delete_mechanism.o insertion_mechanism.o hybrid_cpu.o hybrid_gpu.o pointer_test.o benchmark.o benchmarkLF.o tbPerf.o realistic.o preallocated.o gtest_unordered.o

default: tests
tests: ${OBJ}

clean: 
	rm  ${OBJ}
allclean:
	rm ${OBJ} &
	rm gtestvechost &
	rm gtestvecdevice &
	rm gtestdevicevec &
	rm gtest_hashmap & 
	rm compaction &
	rm compaction2 &
	rm compaction3 &
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

gtest_unordered.o: unordered_set_unit_test/main.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST}  -o gtest_unordered unordered_set_unit_test/main.cu

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

gtest_device_vec.o: gtest_dev_vec/vec_test.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST} -o gtestdevicevec gtest_dev_vec/vec_test.cu

stream_compaction.o: stream_compaction/race.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST} -o compaction stream_compaction/race.cu

preallocated.o: stream_compaction/preallocated.cu
	${CC} --default-stream per-thread  ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST}  -o compaction3 stream_compaction/preallocated.cu

stream_compaction2.o: stream_compaction/unit.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST} -o compaction2 stream_compaction/unit.cu

delete_mechanism.o: delete_by_compaction/main.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST} -o delete_mechanism delete_by_compaction/main.cu

insertion_mechanism.o: insertion_mechanism/main.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST} -o insertion insertion_mechanism/main.cu

pointer_test.o: pointer_test/main.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST} -o pointertest pointer_test/main.cu

hybrid_gpu.o: hybrid/main.cu
	${CC} ${CXXFLAGS} ${OPT} ${EXTRA} ${GTEST} -o hybrid_gpu hybrid/main.cu

hybrid_cpu.o: hybrid/main.cu
	${CC} -L//home/kstppd/libs/googletest/build/lib  -I/home/kstppd/libs/googletest/googletest/include   -DHASHINATOR_CPU_ONLY_MODE  ${CXXFLAGS}    -std=c++17 -o hybrid_cpu hybrid/main.cu   -lgtest -lgtest_main
