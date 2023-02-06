#!/bin/bash
[ -e ".data" ] && rm .data
for i in {10..24};
do
   time=$(nvprof ./benchmark $i 2>&1  | grep insert_kernel  | awk -F " " '{print $6}')
   unit=$(echo $time | tail -c 3)
   time=${time::-3}
   if [ "$unit" = "us"  ]; then 
      result=$(bc -l <<<"${time}/1000")
   else
      result=$(bc -l <<<"${time}")
   fi
   echo ${i} $result >> .data
   echo ${i} $result
   sleep 1
done;


python3 plot.py .data
