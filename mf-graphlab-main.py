# coding: utf-8

# pylint: disable=W0611,C0111,C0103,E1101

import graphlab as gl
import numpy as np
import random
import sys

np.random.seed(123)
random.seed(123)

import gflags

FLAGS = gflags.FLAGS
#gflags.DEFINE_string('train', 'netflix_mm_10000_1000', 'Training File')
#gflags.DEFINE_string('test', 'netflix_mm_10000_1000', 'Testing File')
gflags.DEFINE_string('dataset', 'ratings_debug', 'dataset')
# gflags.DEFINE_string('movie', 'movies', 'Testing File')
gflags.DEFINE_integer('rank', 20, 'Matrix Rank')
gflags.DEFINE_float('lamb', 0.1, 'Lambda')
gflags.DEFINE_float('eta', 0.01, 'Learning Rate')
gflags.DEFINE_integer('maxit', 50, 'Maximum Number of Iterations')
gflags.DEFINE_integer('unified', False, '')
gflags.DEFINE_integer('lamb_w', 0.1, '')
# gflags.DEFINE_integer('rmseint', 5, 'RMSE Computation Interval')
# gflags.DEFINE_integer('cores', -1, 'CPU cores')

from lib import n, m, ng, nht, \
    user_ids, movie_ids, uids, mids, movies, movies_features,\
    load, get_features, prefix, remove, get_vertices, get_graph, \
    rmse_u, rmse, dot_feature, getLR, \
    sgd_triple_updater, sgd_gl_edge, \
    run_full, run_debug, eta_search, \
    search_pure, search_pure_coor, run_pure, \
    search_unified, search_unified_coor, run_mf


def overnightRun():
    # pylint: disable=W0622,W0612
    min_rmse_test, min_lambduh, min_k, min_lambduh_w = search_unified()

    #rmses_cf = run_cf(min_lambduh, min_k, min_lambduh_w)
    #rmses_cf2 = run_cf2(min_lambduh, min_k, min_lambduh_w)


# In[64]:

def main(argv):
    # pylint: disable=W0612
    try:
        argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError, e:
        print '%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS)
        sys.exit(1)

    gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_GRAPH_LAMBDA_WORKERS', 16)
    for flag_name in sorted(FLAGS.RegisteredFlags()):
        if flag_name not in ["?", "help", "helpshort", "helpxml"]:
            fl = FLAGS.FlagDict()[flag_name]
            with open('output/main.out', 'a') as f:
                f.write(
                    "# " + fl.help + " (" + flag_name + "): " + str(fl.value) + '\n')

    X_train, X_test = load(FLAGS.dataset)
    g = get_graph(X_train, FLAGS.rank)

    rmse_train, rmse_test, L, R, wu, wm, bu, bm = \
        sgd_gl_edge(g, X_train, X_test,
                    FLAGS.lamb, FLAGS.rank, FLAGS.eta, Niter=FLAGS.maxit,
                    unified=FLAGS.unified, lambduh_w=FLAGS.lamb_w, output="main")

    print 'rmse_train', rmse_train
    print 'rmse_test', rmse_test

if __name__ == '__main__':
    main(sys.argv)
    # overnightRun()
