import os
import numpy as np

if not os.path.exists("results"):
    os.mkdir("results")

def cmd(st):
    print "cd /projects/grail/supaasorn/LinkPrediction/; " + st
    #os.system(st)
    
lambs = [0, 0.001, 0.01, 0.1, 1, 10]
lambws = [0, 0.001, 0.01, 0.1, 1, 10]
ranks = [5, 10, 20]
for lam in lambs:
    for lamw in lambws:
        for rank in ranks:
            cmd("./nomad -maxit=1 -method=NOMAD -lambda=" + str(lam) + " -lambdaw=" + str(lamw) + " -rank=" + str(rank) + " > output/" + str(rank) + "_" + str(lam) + "_" + str(lamw) + ".txt")
