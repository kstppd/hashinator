import matplotlib
matplotlib.use('Agg')
import os,sys
import matplotlib.pyplot as plt
import numpy as np



Files=sys.argv[1::]
fig, ax1 = plt.subplots()
plt.title("Hashinator Insertion and Retrieval @0.5 Load Factor")
ax1.set_xlabel('Number of Elements')
ax1.set_ylabel("Time for all elements [ms]")
for file in Files:
   data = np.loadtxt(file)
   elems =np.power(2,data[:,0])
   insert =data[:,1]
   ax1.plot(elems, insert, linewidth=0.8, label=file,linestyle='dashdot')


ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.tick_params(axis='y', colors='k')
ax1.grid(which='major', axis='both', linestyle='--')
ax1.legend(loc="upper left")
plt.savefig("bench.png",dpi=500)
