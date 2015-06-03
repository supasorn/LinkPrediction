import re
import os

result = "output_crossvalid"
l = os.listdir(result)

for f in l:
    st = open(result + "/" + f).read()
    print re.search("Lambda LR: (.*)", st).group(1)
    print re.search("Lambda W: (.*)", st).group(1)
    print re.search("Rank: (.*)", st).group(1)
    print re.search("Train RMSE (.*)", st).group(1)
    print re.search("Test RMSE (.*)", st).group(1)
    break
