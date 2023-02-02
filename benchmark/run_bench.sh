#!/bin/bash
[ -e ".data" ] && rm .data
for i in {10..28};
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
done;

python << END
import matplotlib
matplotlib.use('Agg')
import os,sys
import matplotlib.pyplot as plt
import numpy as np

file=".data"

data = np.loadtxt(file)
elems =np.power(2,data[:,0])
insert =data[:,1]


fig, ax1 = plt.subplots()
plt.title("Hashinator Insertion and Retrieval @0.5 Load Factor")
ax1.set_xlabel('Number of Elements')
ax1.set_ylabel("Time for all elements [ms]")
ax1.plot(elems, insert, color='k',linewidth=0.8, label="Insertion",linestyle='dashdot')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.tick_params(axis='y', colors='k')
ax1.grid(which='major', axis='both', linestyle='--')
ax1.legend(loc="upper left")
plt.savefig("bench.png",dpi=500)

END

                                            
