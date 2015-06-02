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
random.seed(123)
np.random.seed(123)


def prefix(p):
    return lambda x: "%s%s"%(p,x)


n, m = (138493, 27278)
ng, nht = 19, 40
user_ids = range(n)
movie_ids = range(m)
uids = map(prefix('u'), user_ids)
mids = map(prefix('m'), movie_ids)

def load(name):
    return gl.load_sframe('data/%s_train.sframe' % name), \
        gl.load_sframe('data/%s_test.sframe' % name)

def get_features(m):
    return dict(zip(m.indices, m.data))


def remove(c):
    def r(x):
        del x[c]
        return x
    return r

def get_vertices(n, m, k, movies_features, factor0=1):
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

def get_graph(X_train, k, movies):
    start = datetime.now()
    factor0 = (X_train['rating'].mean() / k / 0.25) ** 0.5
    vertices  = get_vertices(n, m, k, movies, factor0)
    X_train['uid'] = X_train['userId'].apply(prefix('u'))
    X_train['mid'] = X_train['movieId'].apply(prefix('m'))
    sg = SGraph().add_vertices(vertices, vid_field='__id')\
        .add_edges(X_train, src_field='uid', dst_field='mid')
    print 'get_graph %s' % (datetime.now() - start)
    return sg



def rmse_u(sf, L, R, wu, wm, bu, bm, movies):
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

def rmse(sf, L, R,  wu, wm, bu, bm, movies):
    se = sf.apply(lambda r: (L[r['userId']].dot(R[:,r['movieId']]) - r['rating'])**2)
    return se.mean() ** 0.5


# In[10]:

def dot_feature(wu, wm, features):
    return sum((wu[i] + wm[i]) * x for i, x in features.iteritems())


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

def sgd_triple_updater(eta, lambda_u, lambda_v, unified, lambda_w, biased=False):
    c_u = 1-eta*lambda_u
    c_v = 1-eta*lambda_v
    c_w = 1-eta*lambda_w
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

        if unified or biased:
            src['b'] -= eta_eps
            dst['b'] -= eta_eps

        return (src, edge, dst)
    return updater

def eta_search():
    X_train_debug, X_test_debug = load('ratings_debug')
    min_rmse_test = float('inf')
    min_k, min_lambduh = None, None
    rmse_map = {}
    for eta in [0.01, 0.05, 0.1]:
        print 'eta %s'%eta
        for lambduh in [0.01]: #[0, 0.001, 0.01, 0.1, 1]:
            for k in [5]: #, 10, 20]:
                g = get_graph(X_train_debug, 5, movies_features)
                rmse_train, rmse_test, L, R, wu, wm, bu, bm = \
                    sgd_gl_edge(g, X_train_debug, X_test_debug, lambduh, k, eta, Niter=3)
                rmse_map.get(lambduh, {}).get(k,{})[eta] = rmse_test
                print "l=%s, k=%s, rmse=%.4f" % (lambduh, k, rmse_test[-1][1])
                if rmse_test[-1][1] < min_rmse_test  or np.isnan(rmse_test[-1][1]):
                    min_rmse_test = rmse_test[-1][1]
                    min_k = k
                    min_eta = eta
                    min_lambduh = lambduh
    print min_eta
    return rmse_map, min_lambduh, min_k, min_eta

def run_full():
    X_train, X_test = load('ratings')
    g = get_graph(X_train, 5, movies_features)
    return sgd_gl_edge(g, X_train, X_test, 0.01, 5)

def run_debug(g=None, Niter=1):
    X_train, X_test = load('ratings_debug_small')
    if g is None:
        g = get_graph(X_train, 5, movies_features)
    return sgd_gl_edge(g, X_train, X_test, 0.001, 5, 0.05, Niter=Niter, unified=True, lambduh_w=0.001)
