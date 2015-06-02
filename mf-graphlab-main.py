
# coding: utf-8

# In[1]:

import graphlab as gl
from graphlab import SFrame
from graphlab import SGraph
import numpy as np
from random import sample
import random
import scipy.io as sio
from itertools import izip
from datetime import datetime
from numpy.random import rand
import sys

np.random.seed(123)
random.seed(123)


# In[ ]:

import gflags

FLAGS = gflags.FLAGS
#gflags.DEFINE_string('train', 'netflix_mm_10000_1000', 'Training File')
#gflags.DEFINE_string('test', 'netflix_mm_10000_1000', 'Testing File')
gflags.DEFINE_string('dataset', 'ratings_debug_small', 'dataset')
# gflags.DEFINE_string('movie', 'movies', 'Testing File')
gflags.DEFINE_integer('rank', 10, 'Matrix Rank')
gflags.DEFINE_float('lamb', 0.1, 'Lambda')
gflags.DEFINE_float('eta', 0.01, 'Learning Rate')
gflags.DEFINE_integer('maxit', 50, 'Maximum Number of Iterations')
gflags.DEFINE_integer('unified', False, '')
gflags.DEFINE_integer('lamb_w', 0.1, '')
# gflags.DEFINE_integer('rmseint', 5, 'RMSE Computation Interval')
# gflags.DEFINE_integer('cores', -1, 'CPU cores')

from lib import n, m, ng, nht, user_ids, movie_ids, uids, mids,\
    load, get_features, prefix, remove, get_vertices, get_graph, \
    rmse_u,rmse, dot_feature, getLR, sgd_triple_updater

movies = sio.mmread('data/movies.mtx').tocsr()
movies.eliminate_zeros()
movies_features = list(get_features(movies[i]) for i in xrange(m))


def sgd_gl_edge(g, X_train, X_test,                 lambduh, k, eta=0.05, unified=False, lambduh_w=0, Niter=100, e_rmse=0.005, rmse_train=None):
    get_rmse = rmse_u if unified else rmse

    L, R, wu, wm, bu, bm = getLR(g, unified, k)
    rmse_train = [get_rmse(X_train, L, R, wu, wm, bu, bm) if rmse_train is None else rmse_train]

    print "%s: %.4f" % (0, rmse_train[-1])
    start = datetime.now()

    print 'eta=%s, lambduh=%s, lambduh=%s, unified=%s, lambduh_w=%s' % (eta, lambduh, lambduh, unified, lambduh_w)

    for i in xrange(1, Niter+1):
        g = g.triple_apply(sgd_triple_updater(            eta, lambduh, lambduh, unified, lambduh_w), ['factors', 'w','b'])

        L, R, wu, wm, bu, bm = getLR(g, unified, k)
        rmse_train.append(get_rmse(X_train, L, R, wu, wm, bu, bm))

        print "%s : %.4f (time:%s)" % (i, rmse_train[-1], datetime.now()-start)
        if np.isnan(rmse_train[-1]):
            break

        if abs(rmse_train[-1] - rmse_train[-2]) < e_rmse:
            break

    rmse_test = get_rmse(X_test, L, R, wu, wm, bu, bm)
    print "test=%.4f" % (rmse_test)
    return rmse_train, rmse_test, L, R, wu, wm, bu, bm


def search_pure_mf(eta=0.05):
    X_train_debug, X_test_debug = load('ratings_debug')
    min_rmse_test = float('inf')
    min_k, min_lambduh = None, None
    rmse_map = {}
    for lambduh in [0.001, 0.01, 0.1]:
        for k in [5, 10, 20]:
            g = get_graph(X_train_debug, k, movies)
            rmse_trainunified, rmse_test, L, R, wu, wm, bu, bm = \
                sgd_gl_edge(g, X_train_debug, X_test_debug, lambduh, k, eta, Niter=20)
            rmse_map.get(lambduh, {})[k] = rmse_test
            print "l=%s, k=%s, rmse=%.4f" % (lambduh, k, rmse_test)
            if rmse_test < min_rmse_test:
                min_rmse_test = rmse_test
                min_k = k
                min_lambduh = lambduh
    return min_rmse_test, min_lambduh, min_k


#def search_pure_mf(eta=0.05):
    #X_train_debug, X_test_debug = load('ratings_debug')
    #min_rmse_test = float('inf')
    #min_k, min_lambduh = None, None
    #rmse_map = {}
    #lambduh = 0.01
    #for k in [5, 10, 20]:
        #g = get_graph(X_train_debug, k, movies)
        #rmse_trainunified, rmse_test, L, R, wu, wm, bu, bm =             sgd_gl_edge(g, X_train_debug, X_test_debug, lambduh, k, eta, Niter=20)
        #rmse_map.get(lambduh, {})[k] = rmse_test
        #print "l=%s, k=%s, rmse=%.4f" % (lambduh, k, rmse_test)
        #if rmse_test < min_rmse_test:
            #min_rmse_test = rmse_test
            #min_k = k
            #min_lambduh = lambduh
    #for lambduh in [0.001, 0.1]:
        #g = get_graph(X_train_debug, k, movies)
        #rmse_trainunified, rmse_test, L, R, wu, wm, bu, bm =             sgd_gl_edge(g, X_train_debug, X_test_debug, lambduh, k, eta, Niter=20)
        #rmse_map.get(lambduh, {})[k] = rmse_test
        #print "l=%s, k=%s, rmse=%.4f" % (lambduh, k, rmse_test)
        #if rmse_test < min_rmse_test:
            #min_rmse_test = rmse_test
            #min_lambduh = lambduh
    #return min_rmse_test, min_lambduh, min_k

def run_pure_mf(min_lambduh, min_k, eta=0.05):
    X_train, X_test = load('ratings')
    g = get_graph(X_train, 5, movies)
    rmse_train, rmse_test, L, R =                 sgd_gl_edge(g, X_train, X_test, min_lambduh, min_k)
    print rmse_test
    return rmse_map, rmse_train, rmse_test


# In[ ]:

# optimal_mf = search_pure_mf()


# In[ ]:

# rmse_map, rmse_train, rmse_test = run_pure_mf(min_lambduh, min_k)


# In[ ]:

def search_cf(eta=0.05):
    X_train_debug, X_test_debug = load('ratings_debug_small')
    min_rmse_test = float('inf')
    min_k, min_lambduh, min_lambduh_w = None, None, None
    rmse_map = {}
    for lambduh in [0.001, 0.01, 0.1]:
        for lambduh_w in [0.001, 0.01, 0.1]:
            for k in [5, 10, 20]:
                g = get_graph(X_train_debug, k, movies)
                rmse_train, rmse_test, L, R, wu, wm, bu, bm = sgd_gl_edge(g, X_train_debug, X_test_debug, lambduh, k, eta, unified=True, lambduh_w=lambduh_w)
                rmse_map.get(lambduh, {}).get(k,{})[lambduh_w] = rmse_test
                print "l=%s, k=%s, l_w=%s, rmse=%.4f" % (lambduh, k, lambduh_w, rmse_test)
                if rmse_test < min_rmse_test:
                    min_rmse_test = rmse_test
                    min_k = k
                    min_lambduh = lambduh
                    min_lambduh_w = lambduh_w

    return min_rmse_test, min_lambduh, min_k, min_lambduh_w

def run_cf(min_lambduh, min_k, min_lambduh_w, eta=0.05):
    X_train, X_test = load('ratings')
    g = get_graph(X_train, min_k, movies)
    rmse_train, rmse_test, L, R =                 sgd_gl_edge(g, X_train, X_test, min_lambduh, min_k, min_eta,                                 unified=True, lambduh_w=min_lambduh_w)
    print rmse_test
    return rmse_map, rmse_train, rmse_test


# In[ ]:

# run coldstart to predict cold-start dataset
def run_cf2(min_lambduh, min_k, min_lambduh_w, eta=0.05):
    X_train, X_test = load('ratings_cs')
    g = get_graph(X_train, min_k, movies)
    rmse_train, rmse_test, L, R =                 sgd_gl_edge(g, X_train, X_test, min_lambduh, min_k, min_eta,                                 unified=True, lambduh_w=min_lambduh_w)
    print rmse_test
    return rmse_map, rmse_train, rmse_test


def overnightRun():
    min_rmse_test, min_lambduh, min_k, min_lambduh_w = search_cf()

    #rmses_cf = run_cf(min_lambduh, min_k, min_lambduh_w)
    #rmses_cf2 = run_cf2(min_lambduh, min_k, min_lambduh_w)


# In[64]:

def main(argv):
    try:
      argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError, e:
      print '%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS)
      sys.exit(1)

    gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_GRAPH_LAMBDA_WORKERS', 16)
    for flag_name in sorted(FLAGS.RegisteredFlags()):
        if flag_name not in ["?", "help", "helpshort", "helpxml"]:
            fl = FLAGS.FlagDict()[flag_name]
            print "# " + fl.help + " (" + flag_name + "): " + str(fl.value)

    X_train, X_test = load(FLAGS.dataset)
    g = get_graph(X_train, FLAGS.rank, movies)

    rmse_train, rmse_test, L, R, wu, wm, bu, bm = \
        sgd_gl_edge(g, X_train, X_test,
            FLAGS.lamb, FLAGS.rank, FLAGS.eta, Niter=FLAGS.maxit,\
            unified=FLAGS.unified, lambduh_w=FLAGS.lamb_w)

    print 'rmse_train', rmse_train
    print 'rmse_test', rmse_test

if __name__ == '__main__':
    #main(sys.argv)
    overnightRun()

