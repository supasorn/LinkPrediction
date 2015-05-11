
# coding: utf-8

# In[3]:

from __future__ import division
import scipy as sp
import numpy as np
from scipy import io
import itertools
import math
import time


# In[4]:

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
        


# In[ ]:

def SGD(data, eta = 0.01, lambduh = 0.1, maxit = 10):
    rank = 10
    userOffset = 0
    movieOffset = data.shape[0]
    latent = np.random.rand(sum(data.shape), rank)
    
    it = 0
    innerIt = 0
    cx = data.tocoo() 
    print "Initial RMSE %f" % (RMSE(data, latent))
    while it < maxit:
        start = time.time()
        for user,movie,rating in itertools.izip(cx.row, cx.col, cx.data):
            vMovie = latent[movie + movieOffset]
            vUser = latent[user + userOffset] 
            vUserTmp = vUser.copy()
            
            e = vUser.dot(vMovie) - rating
            c1 = (1 - eta * lambduh)
            
            vUser[:] = c1 * vUser - eta * e * vMovie
            vMovie[:] = c1 * vMovie - eta * e * vUserTmp
            
            # update error
            innerIt += 1
            #if innerIt % 100000 == 0:
        it += 1
        print "time %f" % (time.time() - start)
        print "%d - %f" % (innerIt, RMSE(data, latent))

SGD(data)
# In[ ]:




