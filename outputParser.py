import re
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

#result = "valid_nounified"
result = "valid_unified"
l = os.listdir(result)

def read(st):
    return {
            "lam": re.search("Lambda LR: (.*)", st).group(1),
            "lamw": re.search("Lambda W: (.*)", st).group(1),
            "rank": re.search("Rank: (.*)", st).group(1),
            "train": re.search("Train RMSE :(.*)", st).group(1),
            "test": re.search("Test RMSE :(.*)", st).group(1)}

lambs = [0.001, 0.01, 0.1, 1, 10]
lambws = [0.001, 0.01, 0.1, 1, 10]


bestTest = 1e10
fo2 = open(result + "_validation_result.txt", "w")

for rank in [5, 10, 20]:
    fo2.write("rank = %d\n" % rank)
    for lam in lambs:
        for lamw in lambws:
            st = open(result + "/" + str(rank) + "_" + str(lam) + "_" + str(lamw) + ".txt").read()
            v = read(st)
            #fo.write(v["train"] + ("," if lamw != lambws[-1] else ""))
            fo2.write(v["test"] + ("," if lamw != lambws[-1] else ""))
            if float(v["test"]) < bestTest:
                bestTest = float(v["test"])
                bestRank = rank
                bestLam = lam
                bestLamw = lamw
        fo2.write("\n")
    fo2.write("\n")
    
fo2.write("Best = %f\n" % bestTest)
fo2.write(" rank = %f\n" % bestRank)
fo2.write(" lam = %f\n" % bestLam)
fo2.write(" lamw = %f\n" % bestLamw)
fo2.close()
