from __future__ import division
import scipy as sp
import numpy as np
from scipy import io

def main():
    f = io.mmread("data/netflix_mm")
    print f

if __name__ == "__main__":
    main()
