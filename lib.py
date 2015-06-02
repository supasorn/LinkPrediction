# pylint: disable=C0111,E1101,C0103,W0141,W0110,W0613,R0913,R0914,C0301

import graphlab as gl
from graphlab import SFrame
from graphlab import SGraph
import numpy as np
import random
import scipy.io as sio
from itertools import izip
from datetime import datetime
from numpy.random import rand
random.seed(123)
np.random.seed(123)


def prefix(p):
    return lambda x: "%s%s" % (p, x)


def get_features(movie):
    return dict(zip(movie.indices, movie.data))

n, m = (138493, 27278)
ng, nht = 19, 40
user_ids = range(n)
movie_ids = range(m)
uids = map(prefix('u'), user_ids)
mids = map(prefix('m'), movie_ids)

movies = sio.mmread('data/movies.mtx').tocsr()
movies.eliminate_zeros()
movies_features = list(get_features(movies[i]) for i in xrange(m))


def load(name):
    return gl.load_sframe('data/%s_train.sframe' % name), \
        gl.load_sframe('data/%s_test.sframe' % name)


def remove(c):
    def r(x):
        del x[c]
        return x
    return r


def get_vertices(k, factor0=1):
    return SFrame({
        # Movies
        '__id': mids,
        'factors': map(lambda _: rand(k) * factor0, movie_ids),
        'w': map(lambda _: np.zeros(ng + nht), movie_ids),
        'b': map(lambda _: 0, movie_ids),
        'features': movies_features,
        'user':  map(lambda _: 0, movie_ids)
    }).append(SFrame({
        # User
        '__id': uids,
        'factors': map(lambda _: rand(k) * factor0, user_ids),
        'w': map(lambda _: np.zeros(ng + nht), user_ids),
        'b': map(lambda _: 0, user_ids),
        'features': map(lambda _: {}, user_ids),
        'user': map(lambda _: 1, user_ids)
    }))


def get_graph(X_train, k):
    start = datetime.now()
    factor0 = (X_train['rating'].mean() / k / 0.25) ** 0.5
    vertices = get_vertices(k, factor0)
    X_train['uid'] = X_train['userId'].apply(prefix('u'))
    X_train['mid'] = X_train['movieId'].apply(prefix('m'))
    sg = SGraph().add_vertices(vertices, vid_field='__id')\
        .add_edges(X_train, src_field='uid', dst_field='mid')
    print 'get_graph %s' % (datetime.now() - start)
    return sg


def rmse_u(sf, L, R, wu, wm, bu, bm):
    # pylint: disable=W0621
    def get_se(r):
        u, m, r = r['userId'], r['movieId'], r['rating']
        movie = movies[m]
        rhat = L[u].dot(R[:, m])
        rhat += bu[u] + bm[m]
        rhat += sum((wu[u][i] + wm[m][i]) * x for i,
                    x in izip(movie.indices, movie.data))
#         print 'get_se', rhat, r, (rhat - r) ** 2
        return (rhat - r) ** 2

    se = sf.apply(get_se)
    return se.mean() ** 0.5


def rmse(sf, L, R, wu, wm, bu, bm):
    se = sf.apply(
        lambda r: (L[r['userId']].dot(R[:, r['movieId']]) - r['rating']) ** 2)
    return se.mean() ** 0.5


# In[10]:

def dot_feature(wu, wm, features):
    return sum((wu[i] + wm[i]) * x for i, x in features.iteritems())


def getLR(g, unified, k):
    L = np.ones((n, k))
    R = np.ones((k, m))
    wu = np.zeros((n, ng + nht))
    wm = np.zeros((m, ng + nht))
    bu = np.zeros((n))
    bm = np.zeros((m))

    U = g.get_vertices(fields={'user': 1})
    _uids = np.array(U['__id'].apply(lambda x: x[1:]), dtype=int)
    L[_uids] = np.array(U['factors'])

    M = g.get_vertices(fields={'user': 0})
    _mids = np.array(M['__id'].apply(lambda x: x[1:]), dtype=int)
    R[:, _mids] = np.array(M['factors']).T

    if unified:
        wu[_uids] = np.array(U['w'])
        bu[_uids] = U['b']
        wm[_mids] = np.array(M['w'])
        bm[_mids] = M['b']

    return L, R, wu, wm, bu, bm


def sgd_triple_updater(eta, lambda_u, lambda_v, unified, lambda_w, biased=False):
    c_u = 1 - eta * lambda_u
    c_v = 1 - eta * lambda_v
    c_w = 1 - eta * lambda_w

    def updater(src, edge, dst):
        Lu = np.array(src['factors'])
        Rv = np.array(dst['factors'])
        ruv = edge['rating']
        rhat = Lu.dot(Rv)
        if unified or biased:
            rhat += src['b'] + dst['b']
        if unified:
            rhat += dot_feature(src['w'], dst['w'], dst['features'])

        eps = rhat - ruv
        eta_eps = eta * eps
        src['factors'] = c_u * Lu - eta_eps * Rv
        dst['factors'] = c_v * Rv - eta_eps * Lu
        if unified:
            src['w'] = c_w * np.array(src['w'])
            dst['w'] = c_w * np.array(dst['w'])
            for i, x in dst['features'].iteritems():
                dx = eta_eps * x
                src['w'][i] -= dx
                dst['w'][i] -= dx

        if unified or biased:
            src['b'] -= eta_eps
            dst['b'] -= eta_eps

        return (src, edge, dst)
    return updater


def eta_search():
    #pylint: disable=unused-variable
    X_train_debug, X_test_debug = load('ratings_debug')
    min_rmse_test = float('inf')
    min_k, min_lambduh = None, None
    rmse_map = {}
    for eta in [0.01, 0.05, 0.1]:
        print 'eta %s' % eta
        for lambduh in [0.01]:  # [0, 0.001, 0.01, 0.1, 1]:
            for k in [5]:  # , 10, 20]:
                g = get_graph(X_train_debug, 5)
                (rmse_train, rmse_test, L, R, wu, wm, bu, bm) = \
                    sgd_gl_edge(
                        g, X_train_debug, X_test_debug, lambduh, k, eta, Niter=3)
                rmse_map.get(lambduh, {}).get(k, {})[eta] = rmse_test
                print "l=%s, k=%s, rmse=%.4f" % (lambduh, k, rmse_test[-1][1])
                if rmse_test[-1][1] < min_rmse_test or np.isnan(rmse_test[-1][1]):
                    min_rmse_test = rmse_test[-1][1]
                    min_k = k
                    min_eta = eta
                    min_lambduh = lambduh
    print min_eta
    return rmse_map, min_lambduh, min_k, min_eta


def run_full():
    X_train, X_test = load('ratings')
    g = get_graph(X_train, 5)
    return sgd_gl_edge(g, X_train, X_test, 0.01, 5)


def run_debug(g=None, Niter=1):
    X_train, X_test = load('ratings_debug_small')
    if g is None:
        g = get_graph(X_train, 5)
    return sgd_gl_edge(g, X_train, X_test, 0.001, 5, 0.05, Niter=Niter, unified=True, lambduh_w=0.001, output='debug')


def sgd_gl_edge(g, X_train, X_test,
                lambduh, k, eta=0.05, unified=False, lambduh_w=0, Niter=100,
                e_rmse=0.005, test_freq=5, output=None, save_freq=0, start_i=0):
    # pylint: disable=W0631
    get_rmse = rmse_u if unified else rmse

    def logoutput(s):
        with open('output/%s.out'%output, 'a') as f:
            f.write(s+'\n')
        print s
    def logstdout(s):
        print s

    log = logstdout if output is None else logoutput

    L, R, wu, wm, bu, bm = getLR(g, unified, k)
    rmse_train = [get_rmse(X_train, L, R, wu, wm, bu, bm)]
    rmse_test = []

    start = datetime.now()
    print 'eta=%s, lambduh=%s, unified=%s, lambduh_w=%s\n' % (eta, lambduh, unified, lambduh_w)
    log(" i,  type,   rmse,           time")
    log("%2s, train, %.4f, %s" % (start_i, rmse_train[-1], datetime.now() - start))

    for i in xrange(start_i+1, Niter + 1):
        g = g.triple_apply(sgd_triple_updater(
            eta, lambduh, lambduh, unified, lambduh_w), ['factors', 'w', 'b'])

        L, R, wu, wm, bu, bm = getLR(g, unified, k)
        rmse_train.append(get_rmse(X_train, L, R, wu, wm, bu, bm))

        log("%2s, train, %.4f, %s" % (i, rmse_train[-1], datetime.now() - start))
        if abs(rmse_train[-1] - rmse_train[-2]) < e_rmse or np.isnan(rmse_train[-1]):
            break
        if test_freq > 0 and i % test_freq == 0:
            rmse_test.append((i, get_rmse(X_test, L, R, wu, wm, bu, bm)))
            log("%s,  test, %.4f, %s" % (i, rmse_test[-1][1], datetime.now() - start))

        if save_freq > 0 and i % save_freq == 0:
            g.save('state/%s__%s.graph' % (output, i))

    rmse_test.append((i, get_rmse(X_test, L, R, wu, wm, bu, bm))) # pylint: disable=undefined-loop-variable
    log("%2s,  test, %.4f, %s" % (i, rmse_test[-1][1], datetime.now() - start))
    return rmse_train, rmse_test, L, R, wu, wm, bu, bm


def search_pure(eta=0.05, Niter=20, data='ratings_debug'):
    # pylint: disable=unused-variable
    X_train_debug, X_test_debug = load(data)
    min_rmse_test = float('inf')
    min_k, min_lambduh = None, None
    rmse_map = {}
    for lambduh in [0.001, 0.01, 0.1]:
        for k in [5, 10, 20]:
            g = get_graph(X_train_debug, k)
            rmse_train, rmse_test, L, R, wu, wm, bu, bm = \
                sgd_gl_edge(g, X_train_debug, X_test_debug, lambduh, k, eta, \
                    Niter=Niter, output="search-pure-l_%s-k_%s"%(lambduh, k))
            rmse_map.get(lambduh, {})[k] = rmse_test
            print "l=%s, k=%s, rmse=%.4f" % (lambduh, k, rmse_test[-1][1])
            if rmse_test[-1][1] < min_rmse_test or np.isnan(rmse_test[-1][1]):
                min_rmse_test = rmse_test[-1][1]
                min_k = k
                min_lambduh = lambduh
    return min_rmse_test, min_lambduh, min_k


def search_pure_coor(eta=0.05, Niter=20, data='ratings_debug'):
    # pylint: disable=unused-variable
    X_train_debug, X_test_debug = load(data)
    min_rmse_test = float('inf')
    min_k, min_lambduh = None, None
    rmse_map = {}
    lambduh = 0.01
    for k in [5, 10, 20]:
        g = get_graph(X_train_debug, k)
        rmse_train, rmse_test, L, R, wu, wm, bu, bm = \
            sgd_gl_edge(
                g, X_train_debug, X_test_debug, lambduh, k, eta, \
                    Niter=Niter, output="search-pure-coor-l_%s-k_%s"%(lambduh, k))
        rmse_map.get(lambduh, {})[k] = rmse_test
        print "l=%s, k=%s, rmse=%.4f" % (lambduh, k, rmse_test[-1][1])
        if rmse_test[-1][1] < min_rmse_test or np.isnan(rmse_test[-1][1]):
            min_rmse_test = rmse_test[-1][1]
            min_k = k
            min_lambduh = lambduh
    k = min_k
    for lambduh in [0.001, 0.1]:
        g = get_graph(X_train_debug, k)
        rmse_train, rmse_test, L, R, wu, wm, bu, bm = \
            sgd_gl_edge(
                g, X_train_debug, X_test_debug, lambduh, k, eta, \
                    Niter=Niter, output="search-pure-coor-l_%s-k_%s"%(lambduh, k))
        rmse_map.get(lambduh, {})[k] = rmse_test
        print "l=%s, k=%s, rmse=%.4f" % (lambduh, k, rmse_test[-1][1])
        if rmse_test[-1][1] < min_rmse_test or np.isnan(rmse_test[-1][1]):
            min_rmse_test = rmse_test[-1][1]
            min_lambduh = lambduh
    return min_rmse_test, min_lambduh, min_k


def run_pure(lambduh, k, eta=0.05, Niter=30):
    # pylint: disable=unused-variable

    X_train, X_test = load('ratings')
    g = get_graph(X_train, 5)
    rmse_train, rmse_test, L, R, wu, wm, bu, bm = \
        sgd_gl_edge(g, X_train, X_test, lambduh, k, eta, \
                    Niter=Niter, output="search-pure-coor-l_%s-k_%s"%(lambduh, k))

    print rmse_test[-1][1]
    return rmse_train, rmse_test


def search_unified_coor(eta=0.05, Niter=20, data='ratings_debug'):
    # pylint: disable=unused-variable
    X_train_debug, X_test_debug = load(data)
    min_rmse_test = float('inf')
    min_k, min_lambduh, min_lambduh_w = None, None, None
    rmse_map = {}
    lambduh, lambduh_w = 0.01, 0.01
    for k in [5, 10, 20]:
        g = get_graph(X_train_debug, k)
        rmse_train, rmse_test, L, R, wu, wm, bu, bm = \
            sgd_gl_edge(g, X_train_debug, X_test_debug, lambduh, k, eta,\
                unified=True, lambduh_w=lambduh_w, Niter=Niter, \
                output="pure-l_%s-lw_%s-k_%s"%(lambduh, lambduh_w, k))
        rmse_map.get(lambduh, {}).get(k, {})[lambduh_w] = rmse_test
        print "l=%s, k=%s, l_w=%s, rmse=%.4f" % (lambduh, k, lambduh_w, rmse_test[-1][1])
        if rmse_test[-1][1] < min_rmse_test or np.isnan(rmse_test[-1][1]):
            min_rmse_test = rmse_test[-1][1]
            min_k = k
    k = min_k
    for lambduh in [0.001, 0.01, 0.1]:
        for lambduh_w in [0.001, 0.01, 0.1]:
            g = get_graph(X_train_debug, k)
            rmse_train, rmse_test, L, R, wu, wm, bu, bm = \
                sgd_gl_edge(g, X_train_debug, X_test_debug, lambduh, k, eta,\
                unified=True, lambduh_w=lambduh_w, Niter=Niter, \
                output="pure-l_%s-lw_%s-k_%s"%(lambduh, lambduh_w, k))
            rmse_map.get(lambduh, {}).get(k, {})[lambduh_w] = rmse_test

            print "l=%s, k=%s, l_w=%s, rmse=%.4f" % (lambduh, k, lambduh_w, rmse_test[-1][1])
            if rmse_test[-1][1] < min_rmse_test or np.isnan(rmse_test[-1][1]):
                min_rmse_test = rmse_test[-1][1]
                min_lambduh = lambduh
                min_lambduh_w = lambduh_w

    return min_rmse_test, min_lambduh, min_k, min_lambduh_w


def search_unified(eta=0.05, Niter=20, data='ratings_debug'):
    # pylint: disable=unused-variable
    X_train_debug, X_test_debug = load(data)
    min_rmse_test = float('inf')
    min_k, min_lambduh, min_lambduh_w = None, None, None
    rmse_map = {}
    for k in [5, 10, 20]:
        for lambduh in [0.001, 0.01, 0.1]:
            for lambduh_w in [0.001, 0.01, 0.1]:
                g = get_graph(X_train_debug, k)
                rmse_train, rmse_test, L, R, wu, wm, bu, bm = \
                    sgd_gl_edge(g, X_train_debug, X_test_debug, lambduh, k, eta,\
                        unified=True, lambduh_w=lambduh_w, Niter=Niter, \
                        output="pure-l_%s-lw_%s-k_%s"%(lambduh, lambduh_w, k))
                rmse_map.get(lambduh, {}).get(k, {})[lambduh_w] = rmse_test
                print "l=%s, k=%s, l_w=%s, rmse=%.4f" % (lambduh, k, lambduh_w, rmse_test)
                if rmse_test < min_rmse_test or np.isnan(rmse_test):
                    min_rmse_test = rmse_test
                    min_k = k
                    min_lambduh = lambduh
                    min_lambduh_w = lambduh_w

    return min_rmse_test, min_lambduh, min_k, min_lambduh_w


def run_unified(lambduh, k, lambduh_w, eta=0.05, Niter=30, data='ratings'):
    # pylint: disable=unused-variable
    X_train, X_test = load('ratings')
    g = get_graph(X_train, k)
    rmse_train, rmse_test, L, R, wu, wm, bu, bm = \
        sgd_gl_edge(g, X_train, X_test, lambduh, k, eta, \
            unified=True, lambduh_w=lambduh_w, Niter=Niter, \
            output="pure-l_%s-lw_%s-k_%s"%(lambduh, lambduh_w, k))
    print rmse_test[-1][1]
    return rmse_train, rmse_test

