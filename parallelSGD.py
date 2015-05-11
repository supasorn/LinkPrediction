from __future__ import division
import sys
import scipy as sp
import numpy as np
from scipy import io
import itertools
import math
import time
from multiprocessing import Pool
from scipy.sparse import coo_matrix
import multiprocessing as mp
import ctypes
import random
import gflags

FLAGS = gflags.FLAGS
gflags.DEFINE_string('train', 'netflix_mm_10000_1000', 'Training file')
gflags.DEFINE_string('test', 'netflix_mm_10000_1000', 'Testing file')
gflags.DEFINE_integer('rank', 10, 'Matrix Rank')
gflags.DEFINE_float('lamb', 0.1, 'Lambda')
gflags.DEFINE_float('eta', 0.01, 'Learning rate')
gflags.DEFINE_integer('maxit', 10, 'Maximum iterations')
gflags.DEFINE_integer('rmseint', 5, 'RMSE computation interval')


def RMSEWorker(x):
    global userOffset, movieOffset, mp_arr, latentShape
    r0, c0, data = x
    latent = np.frombuffer(mp_arr.get_obj()).reshape(latentShape)

    cx = data.tocoo() 
    err = 0
    for user,movie,rating in itertools.izip(cx.row, cx.col, cx.data):
        try:
            vMovie = latent[movie + c0 + movieOffset]
            vUser = latent[user + r0 + userOffset]
            err += (vUser.dot(vMovie) - rating) ** 2
        except (KeyboardInterrupt, SystemExit):
            break
    return err

def RMSE2(slices, nnz, p):
    err = 0
    for i in range(len(slices)):
        err += sum(p.map(RMSEWorker, slices[i]))
    return math.sqrt(err / nnz)
        

def RMSE(data, latent):
    return RMSE2(slice(data, mp.cpu_count()), data.nnz, Pool(mp.cpu_count()))

def update(x):
    global userOffset, movieOffset, mp_arr, latentShape, eta, lambduh
    r0, c0, data = x
    latent = np.frombuffer(mp_arr.get_obj()).reshape(latentShape)

    cx = data.tocoo()
    for user,movie,rating in itertools.izip(cx.row, cx.col, cx.data):
        try:
            vMovie = latent[movie + c0 + movieOffset]
            vUser = latent[user + r0 + userOffset] 
            vUserTmp = vUser.copy()
            
            e = vUser.dot(vMovie) - rating
            c1 = (1 - eta * lambduh)
            
            vUser[:] = c1 * vUser - eta * e * vMovie
            vMovie[:] = c1 * vMovie - eta * e * vUserTmp
        except (KeyboardInterrupt, SystemExit):
            break

def slice(data, cores):
    size = data.shape
    splitRow = np.round(np.linspace(0, size[0], cores + 1)).astype(int)
    splitCol = np.round(np.linspace(0, size[1], cores + 1)).astype(int)

    datacsr = data.tocsr()
    rowSlices = [None] * cores
    slices = [[None] * cores for x in range(cores)]

    for i in range(cores):
        rowSlices[i] = datacsr[splitRow[i]:splitRow[i+1],:].tocsc()

    for i in range(cores):
        for j in range(cores):
            colj = (i + j) % cores
            slices[i][j] = (splitRow[j], splitCol[colj], rowSlices[j][:,splitCol[colj]:splitCol[colj+1]])
    return slices



def SGD(data, eta_ = 0.01, lambduh_ = 0.1, rank = 10, maxit = 10):
    global latentShape, userOffset, movieOffset, mp_arr, eta, lambduh
    t1 = time.time()
    eta = eta_
    lambduh = lambduh_
    userOffset = 0
    movieOffset = data.shape[0]
   
    # Allocate shared memory across processors for latent variable 
    latentShape = (sum(data.shape), rank)
    mp_arr = mp.Array(ctypes.c_double, latentShape[0] * latentShape[1])
    latent = np.frombuffer(mp_arr.get_obj()).reshape(latentShape)

    # Initialize latent variable so that expectation equals average rating
    avgRating = data.sum() / data.nnz
    latent[:] = np.random.rand(sum(data.shape), rank) * math.sqrt(avgRating / rank / 0.25)

    cores = mp.cpu_count()
    slices = slice(data, cores)

    p = Pool(cores)
    it = 0
    print "Initial RMSE %f" % (RMSE2(slices, data.nnz, p))
    while it < maxit:
        start = time.time()

        for i in range(len(slices)):
            p.map(update, slices[i])
        it += 1

        print "%d : time %f : RMSE %s " % (it, time.time() - start, "[NE]" if it % FLAGS.rmseint else str(RMSE2(slices, data.nnz, p)))

    print "Total training time %f" % (time.time() - t1)
    return latent


def main(argv):
    try:
      argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError, e:
      print '%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS)
      sys.exit(1)

    random.seed(1)
    np.random.seed(1)

    print "Training Dataset : %s" % FLAGS.train
    print "Testing Dataset : %s" % FLAGS.test

    dataTraining = io.mmread("data/" + FLAGS.train)
    dataTesting = io.mmread("data/" + FLAGS.test)

    latent = SGD(dataTraining, FLAGS.eta, FLAGS.lamb, FLAGS.rank, FLAGS.maxit)
    print "RMSE Train %f" % (RMSE(dataTraining, latent))
    print "RMSE Test  %f" % (RMSE(dataTesting, latent))

if __name__ == '__main__':
    main(sys.argv)
    



