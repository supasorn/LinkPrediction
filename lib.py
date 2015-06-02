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

def load(name):
    return gl.load_sframe('data/%s_train.sframe' % name), \
        gl.load_sframe('data/%s_test.sframe' % name)

def get_features(m):
    return dict(zip(m.indices, m.data))


def prefix(p):
    return lambda x: "%s%s"%(p,x)

def remove(c):
    def r(x):
        del x[c]
        return x
    return r

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
