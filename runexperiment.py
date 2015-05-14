import os
import numpy as np

if not os.path.exists("results"):
    os.mkdir("results")

    
lambs = np.linspace(0, 0.2, 21)
fo = open("sungrid/cmd1.txt", "w")
count = 0
for lam in lambs:
    fo.write("cd /projects/grail/supasorn/LinkPrediction/; ")
    fo.write("python2.7 parallelSGD.py --train=ratings_debug_train.mtx --test=ratings_debug_test.mtx --lamb=%f --rmseint=1 --maxit=300 > results/%04d.txt\n" % (lam, count))
    count += 1
    print lam
fo.close()

os.system("cd /projects/grail/supasorn/LinkPrediction/sungrid/; python qsub.py cmd1.txt")
