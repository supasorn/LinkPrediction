
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


# In[2]:

def load(name):
    return gl.load_sframe('data/%s_train.sframe' % name),         gl.load_sframe('data/%s_test.sframe' % name)


# In[52]:

def get_features(m):
    return dict(zip(m.indices, m.data))


# In[5]:

def prefix(p):
    return lambda x: "%s%s"%(p,x)

def remove(c):
    def r(x):
        del x[c]
        return x
    return r


# In[54]:

n, m = (138493, 27278)
ng, nht = 19, 40 #
user_ids = range(n)
movie_ids = range(m)
uids = map(prefix('u'), user_ids)
mids = map(prefix('m'), movie_ids)


# In[57]:

movies = sio.mmread('data/movies.mtx').tocsr()
movies.eliminate_zeros()
movies_features = list(get_features(movies[i]) for i in xrange(m))


# In[62]:

def get_vertices(n, m, k, movies, factor0=1):
    return SFrame({
            # Movies
            '__id': mids,
            'factors': map(lambda _: rand(k) * factor0, movie_ids),
            'w': map(lambda _: np.zeros(ng+nht), movie_ids),
            'b': map(lambda _: 0, movie_ids),
            'features': movies_features,
            'user':  map(lambda _: 0, movie_ids)
        }).append(SFrame({
            # User
            '__id': uids,
            'factors': map(lambda _: rand(k) * factor0, user_ids),
            'w': map(lambda _: np.zeros(ng+nht), user_ids),
            'b': map(lambda _: 0, user_ids),
            'features': map(lambda _:{}, user_ids),
            'user': map(lambda _: 1, user_ids)
        }))


# In[8]:

def get_graph(X_train, k, movies):
    factor0 = (X_train['rating'].mean() / k / 0.25) ** 0.5
    vertices  = get_vertices(n, m, k, movies, factor0)
    X_train['uid'] = X_train['userId'].apply(prefix('u'))
    X_train['mid'] = X_train['movieId'].apply(prefix('m'))
    return SGraph().add_vertices(vertices, vid_field='__id')        .add_edges(X_train, src_field='uid', dst_field='mid')


# In[9]:

def rmse_u(sf, L, R, wu, wm, bu, bm):
    def get_se(r):
        u, m, r = r['userId'], r['movieId'], r['rating']
        movie = movies[m]
        rhat = L[u].dot(R[:,m])
        rhat += bu[u] + bm[m]
        rhat += sum((wu[u][i]+wm[m][i])* x for i, x in izip(movie.indices, movie.data))
#         print 'get_se', rhat, r, (rhat - r) ** 2
        return (rhat - r) ** 2

    se = sf.apply(get_se)
    return se.mean() ** 0.5

def rmse(sf, L, R,  wu, wm, bu, bm):
    se = sf.apply(lambda r: (L[r['userId']].dot(R[:,r['movieId']]) - r['rating'])**2)
    return se.mean() ** 0.5


# In[10]:

def dot_feature(wu, wm, features):
    return sum((wu[i] + wm[i]) * x for i, x in features.iteritems())


# In[11]:

def getLR(g, unified, k):
    L = np.ones((n, k))
    R = np.ones((k, m))
    wu = np.zeros((n, ng+nht))
    wm = np.zeros((m, ng+nht))
    bu = np.zeros((n))
    bm = np.zeros((m))

    U = g.get_vertices(fields={'user':1})
    uids = np.array(U['__id'].apply(lambda x: x[1:]), dtype=int)
    L[uids] = np.array(U['factors'])

    M = g.get_vertices(fields={'user':0})
    mids = np.array(M['__id'].apply(lambda x: x[1:]),dtype=int)
    R[:,mids] = np.array(M['factors']).T

    if unified:
        wu[uids] = np.array(U['w'])
        bu[uids] = U['b']
        wm[mids] = np.array(M['w'])
        bm[mids] = M['b']

    return L, R, wu, wm, bu, bm


# In[290]:

# updater = sgd_triple_updater(eta=0.05, lambda_u=0.01, lambda_v=0.01, unified=True, lambda_w=.01)
# e = g.get_edges()[0]
# src = g.get_vertices(ids=e['__src_id'])
# dst = g.get_vertices(ids=e['__dst_id'])
# updater(src, e, dst)


# In[49]:

def sgd_triple_updater(eta, lambda_u, lambda_v, unified, lambda_w):
    c_u = 1-eta*lambda_u
    c_v = 1-eta*lambda_v
    c_w = 1-eta*lambda_w
    def updater(src, edge, dst):
        Lu = np.array(src['factors'])
        Rv = np.array(dst['factors'])
        ruv = edge['rating']
        rhat = Lu.dot(Rv)
        if unified:
            rhat += src['b'] + dst['b'] + dot_feature(src['w'], dst['w'], dst['features'])

        eps = rhat - ruv
        eta_eps = eta*eps
        src['factors'] = c_u * Lu - eta_eps * Rv
        dst['factors'] = c_v * Rv - eta_eps * Lu
        if unified:
            src['w'] = c_w * np.array(src['w'])
            dst['w'] = c_w * np.array(dst['w'])
            for i, x in dst['features'].iteritems():
                dx = eta_eps * x
                src['w'][i] -= dx
                dst['w'][i] -= dx

            src['b'] -= eta_eps
            dst['b'] -= eta_eps

        return (src, edge, dst)
    return updater


# In[13]:

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



# In[14]:

def run_full():
    X_train, X_test = load('ratings')
    g = get_graph(X_train, 5, movies)
    return sgd_gl_edge(g, X_train, X_test, 0.01, 5)


# In[15]:

def run_debug(g=None, Niter=1):
    X_train, X_test = load('ratings_debug_small')
    if g is None:
        g = get_graph(X_train, 5, movies)
    return sgd_gl_edge(g, X_train, X_test, 0.1, 5, 0.01, Niter=Niter, unified=True, lambduh_w=0.1)


# In[63]:

# rmse_train, rmse_test, L, R, wu, wm, bu, bm = run_debug()


# In[ ]:

def eta_search():
    X_train_debug, X_test_debug = load('ratings_debug')
    min_rmse_test = float('inf')
    min_k, min_lambduh = None, None
    rmse_map = {}
    for eta in [0.01, 0.05, 0.1]:
        print 'eta %s'%eta
        for lambduh in [0.01]: #[0, 0.001, 0.01, 0.1, 1]:
            for k in [5]: #, 10, 20]:
                g = get_graph(X_train_debug, 5, movies)
                rmse_train, rmse_test, L, R, wu, wm, bu, bm =                     sgd_gl_edge(g, X_train_debug, X_test_debug, lambduh, k, eta, Niter=3)
                rmse_map.get(lambduh, {}).get(k,{})[eta] = rmse_test
                print "l=%s, k=%s, rmse=%.4f" % (lambduh, k, rmse_test)
                if rmse_test < min_rmse_test:
                    min_rmse_test = rmse_test
                    min_k = k
                    min_eta = eta
                    min_lambduh = lambduh
    print min_eta
    return rmse_map, min_lambduh, min_k, min_eta


# In[67]:

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
    X_train_debug, X_test_debug = load('ratings_debug')
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

