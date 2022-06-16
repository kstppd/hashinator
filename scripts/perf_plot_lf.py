import numpy as np
import sys
import matplotlib.pyplot as plt



file=sys.argv[1]
data=np.loadtxt(file)
time=data[:,0]
lf=data[:,1]

elems=16777216
time_per_billion=[]

for t in time:
    tpb=t*1e9/elems
    print(t,tpb)
    time_per_billion.append(1e9/t*1e9/elems/1e12)

plt.figure(1)
plt.scatter(lf,time_per_billion,s=3,c="k")
plt.title("Insertion rate")
plt.xlabel("Load factor [%]")
plt.ylabel("Billion operations per second")
plt.grid()
plt.savefig("stats_lf.png",dpi=300)
