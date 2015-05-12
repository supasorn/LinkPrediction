import sys
import os

if len(sys.argv) == 2:
    f = open(sys.argv[1], "r")
    num = 0
    
    for l in f:
        fo = open("sh/s%04d.sh" % (num), "w")
        fo.write("#!/usr/bin/env sh\n")
        fo.write(l)
        fo.close()
        num += 1

    for i in range(num):
        st = "qsub -q all.q sh/s%04d.sh" % i
        print st
        os.system(st)
