from __future__ import division
import sys
import scipy as sp
import numpy as np
from scipy import io
import itertools
import math
import time
from multiprocessing import Pool, Queue
from scipy.sparse import coo_matrix
import multiprocessing as mp
import ctypes
import random
import gflags
import os
import copy

FLAGS = gflags.FLAGS
#gflags.DEFINE_string('train', 'netflix_mm_10000_1000', 'Training File')
#gflags.DEFINE_string('test', 'netflix_mm_10000_1000', 'Testing File')
gflags.DEFINE_string('train', 'ratings_debug_train', 'Training File')
gflags.DEFINE_string('test', 'ratings_debug_test', 'Testing File')
gflags.DEFINE_string('movie', 'movies', 'Testing File')
gflags.DEFINE_integer('rank', 10, 'Matrix Rank')
gflags.DEFINE_float('lamb', 0.1, 'Lambda')
gflags.DEFINE_float('lambw', 0.1, 'Lambda')
gflags.DEFINE_float('eta', 0.01, 'Learning Rate')
gflags.DEFINE_integer('maxit', 50, 'Maximum Number of Iterations')
gflags.DEFINE_integer('rmseint', 5, 'RMSE Computation Interval')
gflags.DEFINE_integer('cores', -1, 'CPU cores')
gflags.DEFINE_bool("unified", False, 'unified')


def RMSEWorker(x):
    global userOffset, movieOffset, mp_arr, mp_w, mp_b, latentShape, weightShape, biasShape, movies

    #movies = io.mmread("data/" + FLAGS.movie).tocsr()

    r0, c0, data = x
    latent = np.frombuffer(mp_arr.get_obj()).reshape(latentShape)
    if FLAGS.unified:
        weights = np.frombuffer(mp_w.get_obj()).reshape(weightShape)
        biases = np.frombuffer(mp_b.get_obj()).reshape(biasShape)

    cx = data.tocoo() 
    err = 0
    for user,movie,rating in itertools.izip(cx.row, cx.col, cx.data):
        try:
            mid = c0 + movie + movieOffset
            uid = r0 + user + userOffset
            vMovie = latent[mid]
            vUser = latent[uid]
            pred = vUser.dot(vMovie)
            if FLAGS.unified: 
                wMovie = weights[mid]
                wUser = weights[uid]
                bMovie = biases[mid] 
                bUser = biases[uid]
                pred += bMovie + bUser 

                mov = movies[c0 + movie]
                for j, fea in itertools.izip(mov.indices, mov.data):
                    pred += (wMovie[j] + wUser[j]) * fea

            err += (pred - rating) ** 2
        except (KeyboardInterrupt, SystemExit):
            break
    return err

def RMSE2(slices, nnz, p):
    err = 0
    for i in range(len(slices)):
        err += sum(p.map(RMSEWorker, slices[i]))
    return math.sqrt(err / nnz)
        

def RMSE(data, latent):
    return RMSE2(slice(data, FLAGS.cores), data.nnz, Pool(FLAGS.cores))

def rowSlice(data, cores):
    size = data.shape
    splitRow = np.round(np.linspace(0, size[0], cores + 1)).astype(int)
    datacsr = data.tocsr()
    slices = [None] * cores
    for i in range(cores):
        slices[i] = (i, splitRow[i], datacsr[splitRow[i]:splitRow[i+1],:].tocsc())

    return slices


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

def printLog(it, time, ttime, rmse):
    print "@ %d : [%.3fs, %.3fs] : %s" % (it, time, ttime, rmse)

def update(x):
    global userOffset, movieOffset, mp_arr, mp_w, mp_b, latentShape, weightShape, biasShape, eta, lambduh, lambduh_w, movies

    r0, c0, data = x
    latent = np.frombuffer(mp_arr.get_obj()).reshape(latentShape)
    weights = np.frombuffer(mp_w.get_obj()).reshape(weightShape)
    biases = np.frombuffer(mp_b.get_obj()).reshape(biasShape)

    cx = data.tocoo()
    for user,movie,rating in itertools.izip(cx.row, cx.col, cx.data):
        try:
            mid = movie + c0 + movieOffset
            uid = user + r0 + userOffset

            vMovie = latent[mid]
            vUser = latent[uid] 
            vUserTmp = vUser.copy()
            c1 = (1 - eta * lambduh)

            e = vUser.dot(vMovie)
            if FLAGS.unified:
                wMovie = weights[mid]
                wUser = weights[uid]
                bMovie = biases[mid] 
                bUser = biases[uid]

                e += bMovie + bUser
                mov = movies[c0 + movie]
                for j, fea in itertools.izip(mov.indices, mov.data):
                    e += (wMovie[j] + wUser[j]) * fea

            e -= rating
            
            vUser[:] = c1 * vUser - eta * e * vMovie
            vMovie[:] = c1 * vMovie - eta * e * vUserTmp
            
            if FLAGS.unified:
                c2 = (1 - eta * lambduh_w)
                wMovie[:] = c2 * wMovie
                wUser[:] = c2 * wUser

                for j, fea in itertools.izip(mov.indices, mov.data):
                    t = eta * e * fea
                    wMovie[j] -= t
                    wUser[j] -= t

                bMovie[:] -= eta * e
                bUser[:] -= eta * e

        except (KeyboardInterrupt, SystemExit):
            break

def SGD(data, movies_, eta_ = 0.01, lambduh_ = 0.1, lambduh_w_ = 0.1, rank = 10, maxit = 10):
    global latentShape, userOffset, movieOffset, mp_arr, mp_w, mp_b, biasShape, weightShape, eta, lambduh, movies, lambduh_w
    movies = movies_.tocsr()
    t1 = time.time()
    eta = eta_
    lambduh = lambduh_
    lambduh_w = lambduh_w_
    userOffset = 0
    movieOffset = data.shape[0]
   
    # Allocate shared memory across processors for latent variable 
    latentShape = (sum(data.shape), rank)
    mp_arr = mp.Array(ctypes.c_double, latentShape[0] * latentShape[1])
    latent = np.frombuffer(mp_arr.get_obj()).reshape(latentShape)

    weightShape = (latentShape[0], movies.shape[1])
    mp_w = mp.Array(ctypes.c_double, weightShape[0] * weightShape[1])
    weights = np.frombuffer(mp_w.get_obj()).reshape(weightShape)

    biasShape = (latentShape[0], 1)
    mp_b = mp.Array(ctypes.c_double, biasShape[0] * biasShape[1])
    biases = np.frombuffer(mp_b.get_obj()).reshape(biasShape)

    # Initialize latent variable so that expectation equals average rating
    avgRating = data.sum() / data.nnz
    latent[:] = np.random.rand(sum(data.shape), rank) * math.sqrt(avgRating / rank / 0.25)
    weights[:] = np.zeros(weightShape)
    biases[:] = np.zeros(biasShape)

    slices = slice(data, FLAGS.cores)

    p = Pool(FLAGS.cores)
    it = 0
    printLog(0, 0, time.time() - t1, RMSE2(slices, data.nnz, p))
    while it < maxit:
        start = time.time()

        for i in range(len(slices)):
            p.map(update, slices[i])
        it += 1

        printLog(it, time.time() - start, time.time() - t1, "[NE]" if it % FLAGS.rmseint else str(RMSE2(slices, data.nnz, p)))
    return latent


def updateNOMAD(x):
    global userOffset, movieOffset, mp_arr, mp_w, mp_b, latentShape, weightShape, biasShape, eta, lambduh, lambduh_w, counter, qsize, movies

    #movies = io.mmread("data/" + FLAGS.movie).tocsr()

    i, r0, data, qs = x
    latent = np.frombuffer(mp_arr.get_obj()).reshape(latentShape)
    weights = np.frombuffer(mp_w.get_obj()).reshape(weightShape)
    biases = np.frombuffer(mp_b.get_obj()).reshape(biasShape)


    while True:
        #print [x for x in qsize]
        #print [x.qsize() for x in qs]
        while not qs[i].empty():
            col = qs[i].get()
            column = data[:, col[0]:col[1]].tocoo()
            for user, movie, rating in itertools.izip(column.row, column.col, column.data):
                #print user, movie
                mid = col[0] + movie + movieOffset
                uid = user + r0 + userOffset
                vMovie = latent[mid]
                vUser = latent[uid] 
                vUserTmp = vUser.copy()
                c1 = (1 - eta * lambduh)

                e = vUser.dot(vMovie)
                if FLAGS.unified:
                    wMovie = weights[mid]
                    wUser = weights[uid]
                    bMovie = biases[mid] 
                    bUser = biases[uid]

                    e += bMovie + bUser
                    mov = movies[col[0] + movie]
                    for j, fea in itertools.izip(mov.indices, mov.data):
                        e += (wMovie[j] + wUser[j]) * fea

                e -= rating
                
                vUser[:] = c1 * vUser - eta * e * vMovie
                vMovie[:] = c1 * vMovie - eta * e * vUserTmp
                
                if FLAGS.unified:
                    c2 = (1 - eta * lambduh_w)
                    wMovie[:] = c2 * wMovie
                    wUser[:] = c2 * wUser

                    for j, fea in itertools.izip(mov.indices, mov.data):
                        t = eta * e * fea
                        wMovie[j] -= t
                        wUser[j] -= t

                    bMovie[:] -= eta * e
                    bUser[:] -= eta * e

            with counter.get_lock():
                counter.value += 1

            #nex = np.random.randint(0, len(qsize))
            nex = np.argmin(qsize)
            qs[nex].put(col)

            with qsize.get_lock():
                qsize[nex] += 1
                qsize[i] -= 1

    

def SGDNOMAD(data, movies_, eta_ = 0.01, lambduh_ = 0.1, lambduh_w_ = 0.1, rank = 10, maxit = 10):
    global latentShape, weightShape, biasShape, userOffset, movieOffset, mp_arr, mp_w, mp_b, eta, lambduh, lambduh_w, counter, qsize, movies
    movies = movies_.tocsr()
    t1 = time.time()
    eta = eta_
    lambduh = lambduh_
    lambduh_w = lambduh_w_
    userOffset = 0
    movieOffset = data.shape[0]
   
    # Allocate shared memory across processors for latent variable 
    latentShape = (sum(data.shape), rank)
    mp_arr = mp.Array(ctypes.c_double, latentShape[0] * latentShape[1])
    latent = np.frombuffer(mp_arr.get_obj()).reshape(latentShape)

    weightShape = (latentShape[0], movies.shape[1])
    mp_w = mp.Array(ctypes.c_double, weightShape[0] * weightShape[1])
    weights = np.frombuffer(mp_w.get_obj()).reshape(weightShape)

    biasShape = (latentShape[0], 1)
    mp_b = mp.Array(ctypes.c_double, biasShape[0] * biasShape[1])
    biases = np.frombuffer(mp_b.get_obj()).reshape(biasShape)

    counter = mp.Value('i', 0)
    qsize = mp.Array('i', [0] * FLAGS.cores)

    # Initialize latent variable so that expectation equals average rating
    avgRating = data.sum() / data.nnz
    latent[:] = np.random.rand(latentShape[0], latentShape[1]) * math.sqrt(avgRating / rank / 0.25)
    weights[:] = np.zeros(weightShape)
    biases[:] = np.zeros(biasShape)

    slices = slice(data, FLAGS.cores)
    rowSlices = rowSlice(data, FLAGS.cores)

    p2 = Pool(FLAGS.cores)
    p = Pool(FLAGS.cores)
    it = 0
    printLog(0, 0, time.time() - t1, RMSE2(slices, data.nnz, p2))


    manager = mp.Manager()
    queues = [manager.Queue() for x in range(FLAGS.cores)]

    colList = np.round(np.linspace(0, data.shape[1], (FLAGS.cores) * 20 + 1)).astype(int)

    #for i in range(data.shape[1]):
        #queues[np.random.randint(0, FLAGS.cores)].put(i)

    for i in range(len(colList) - 1):
        r = np.random.randint(0, FLAGS.cores)
        queues[r].put((colList[i], colList[i+1]))
        qsize[r] += 1

    p.map_async(updateNOMAD, [(i, a, b, queues) for i, a, b in rowSlices])

    countPerEpoch = FLAGS.cores * (len(colList) - 1)
    start = time.time()
    #print [q.qsize() for q in queues]
    print [q for q in qsize]
    print "countPerEpoch %d" % countPerEpoch
    while counter.value < countPerEpoch * 300: 
        time.sleep(60 * 3)
        printLog(it, time.time() - start, 0, RMSE2(slices, data.nnz, p2))
        print counter.value
        #print sum([q.qsize() for q in queues])
        print [q for q in qsize]

    print "done"

    p.close()
    p.join()

    return latent

def main(argv):
    try:
      argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError, e:
      print '%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS)
      sys.exit(1)
    os.system("taskset -p 0xFF %d" % os.getpid())

    if FLAGS.cores == -1:
        FLAGS.cores = mp.cpu_count()

    for flag_name in sorted(FLAGS.RegisteredFlags()):
        if flag_name not in ["?", "help", "helpshort", "helpxml"]:
            fl = FLAGS.FlagDict()[flag_name]
            print "# " + fl.help + " (" + flag_name + "): " + str(fl.value)
    random.seed(1)
    np.random.seed(1)

    dataTraining = io.mmread("data/" + FLAGS.train)
    dataTesting = io.mmread("data/" + FLAGS.test)
    movies = io.mmread("data/" + FLAGS.movie)
    print dataTraining.shape
    print dataTesting.shape
    
    print movies.shape



    latent = SGDNOMAD(dataTraining, movies, FLAGS.eta, FLAGS.lamb, FLAGS.lambw, FLAGS.rank, FLAGS.maxit)
    #latent = SGD(dataTraining, movies, FLAGS.eta, FLAGS.lamb, FLAGS.lambw, FLAGS.rank, FLAGS.maxit)


if __name__ == '__main__':
    main(sys.argv)
    


