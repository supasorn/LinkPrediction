import os
import numpy as np

def cmd(st):
    print "cd /projects/grail/supasorn/LinkPrediction/; " + st
    #os.system(st)
    
lambs = [0.001, 0.01, 0.1, 1, 10]
lambws = [0.001, 0.01, 0.1, 1, 10]
ranks = [5, 10, 20]
for lam in lambs:
    for lamw in lambws:
        for rank in ranks:
            #cmd("./nomad -nounified -maxit=100 -byit -method=DSGD -lambda=" + str(lam) + " -lambdaw=" + str(lamw) + " -rank=" + str(rank) + " > output/" + str(rank) + "_" + str(lam) + "_" + str(lamw) + ".txt")
            cmd("./nomad -nounified -maxit=100 -byit -method=DSGD -lambda=" + str(lam) + " -lambdaw=" + str(lamw) + " -rank=" + str(rank) + " -data=data/ratings_debug_train.mtx -datatest=data/ratings_debug_test.mtx -movie=data/movies_ratings_debug.mtx > valid_nounified/" + str(rank) + "_" + str(lam) + "_" + str(lamw) + ".txt")
