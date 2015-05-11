
# coding: utf-8

# In[3]:

from __future__ import division
import scipy as sp
import numpy as np
from scipy import io
import itertools
import math
import time
import multiprocessing
from multiprocessing import Lock
from multiprocessing.pool import ThreadPool
from collections import deque
from scipy.sparse import coo_matrix


lock = Lock()
data = io.mmread("data/netflix_mm_10000_1000")
data.shape


# In[18]:

def RMSE(data, latent):
    userOffset = 0
    movieOffset = data.shape[0]
    
    cx = data.tocoo() 
    err = 0
    for user,movie,rating in itertools.izip(cx.row, cx.col, cx.data):
        vUser = latent[user + userOffset]
        vMovie = latent[movie + movieOffset]
        err += (vUser.dot(vMovie) - rating) ** 2
        #print "%f %f" % (vUser.dot(vMovie), rating)
    return math.sqrt(err / data.nnz)
    #return err
        

def update(x):
    global latent, userOffset, movieOffset, lock
    r0, c0, data, eta, lambduh = x

    cx = data.tocoo()

    p = True
    oldLatent = latent.copy()
    for user,movie,rating in itertools.izip(cx.row, cx.col, cx.data):
        vMovie = latent[movie + c0 + movieOffset]
        vUser = latent[user + r0 + userOffset] 
        vUserTmp = vUser.copy()
        
        e = vUser.dot(vMovie) - rating
        c1 = (1 - eta * lambduh)
        
        vUser[:] = c1 * vUser - eta * e * vMovie
        vMovie[:] = c1 * vMovie - eta * e * vUserTmp
        #if p:
            #with lock:
                #print vUser, vUserTmp, latent[user + r0 + userOffset]
                #print "==="
            #p = False
    #print np.sum(oldLatent - latent)

# In[ ]:

def SGD(data, eta = 0.01, lambduh = 0.1, maxit = 2):
    global latent, userOffset, movieOffset
    rank = 10
    userOffset = 0
    movieOffset = data.shape[0]
    latent = np.random.rand(sum(data.shape), rank)

    size = data.shape
    cores = 4 
    splitRow = np.round(np.linspace(0, size[0], cores + 1)).astype(int)
    splitCol = np.round(np.linspace(0, size[1], cores + 1)).astype(int)

    p = ThreadPool(cores)

    it = 0
    print "Initial RMSE %f" % (RMSE(data, latent))

    datacsr = data.tocsr()
    rowSlices = [None] * cores
    slices = [None] * cores
    for i in range(cores):
        rowSlices[i] = datacsr[splitRow[i]:splitRow[i+1],:].tocsc()

    while it < maxit:
        oldLatent = latent.copy()
        start = time.time()


        for i in range(cores):
            for j in range(cores):
                colj = (i + j) % cores
                slices[j] = (splitRow[j], splitCol[colj], rowSlices[j][:,splitCol[colj]:splitCol[colj+1]], eta, lambduh)
            p.map(update, slices)
        it += 1
        print "time %f" % (time.time() - start)
        print "%f" % (RMSE(data, latent))
        print np.sum(oldLatent - latent)

SGD(data)

