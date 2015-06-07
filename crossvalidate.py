import os
import numpy as np


pref = "cd /projects/grail/supasorn/LinkPrediction/; " 
    
f1 = open("cmd_nounified.txt", "w")
f2 = open("cmd_unified.txt", "w")
lambs = [0.001, 0.01, 0.1, 1, 10]
lambws = [0.001, 0.01, 0.1, 1, 10]
ranks = [5, 10, 20]
for lam in lambs:
    for lamw in lambws:
        for rank in ranks:
            f1.write(pref + "./nomad -onermse -maxit=100 -byit -method=DSGD -lambda=" + str(lam) + " -lambdaw=" + str(lamw) + " -rank=" + str(rank) + " -nounified > valid_nounified/" + str(rank) + "_" + str(lam) + "_" + str(lamw) + ".txt\n")

            f2.write(pref + "./nomad -onermse -maxit=100 -byit -method=DSGD -lambda=" + str(lam) + " -lambdaw=" + str(lamw) + " -rank=" + str(rank) + " -unified > valid_unified/" + str(rank) + "_" + str(lam) + "_" + str(lamw) + ".txt\n")

f1.close()
f2.close()
