import re
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

result = "output_crossvalid_nounified"
result = "output_crossvalid"
l = os.listdir(result)

def read(st):
    return {
            "lam": re.search("Lambda LR: (.*)", st).group(1),
            "lamw": re.search("Lambda W: (.*)", st).group(1),
            "rank": re.search("Rank: (.*)", st).group(1),
            "train": re.search("Train RMSE (.*)", st).group(1),
            "test": re.search("Test RMSE (.*)", st).group(1)}

lambs = [0, 0.001, 0.01, 0.1, 1, 10]
lambws = [0, 0.001, 0.01, 0.1, 1, 10]


bestTest = 1e10
for rank in [5, 10, 20]:
    fo = open(result + "_rank" + str(rank) + "_train.txt", "w")
    fo2 = open(result + "_rank" + str(rank) + "_test.txt", "w")
    for lam in lambs:
        for lamw in lambws:
            st = open(result + "/" + str(rank) + "_" + str(lam) + "_" + str(lamw) + ".txt").read()
            v = read(st)
            fo.write(v["train"] + ("," if lamw != lambws[-1] else ""))
            fo2.write(v["test"] + ("," if lamw != lambws[-1] else ""))
            if float(v["test"]) < bestTest:
                bestTest = float(v["test"])
        fo.write("\n")
        fo2.write("\n")
    fo.close()
    fo2.close()
    
print "Best = %f" % bestTest
exit(0)
fig = plt.figure()
ax = fig.gca(projection='3d')

lambs = [0, 0.001, 0.01, 0.1, 1, 10]
lambws = [0, 0.001, 0.01, 0.1, 1, 10]

#X = np.arange(-5, 5, 0.25)
#Y = np.arange(-5, 5, 0.25)
X = np.array(lambs)
Y = np.array(lambs)
X, Y = np.meshgrid(X, Y)

#R = np.sqrt(X**2 + Y**2)
#Z = np.sin(R)
#print Z
Z = np.ones((len(X), len(Y))) * 3
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=True)
ax.set_zlim(-1.01, 1.01)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

