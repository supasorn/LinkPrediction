import scipy.io as sio
import pandas as pd
from scipy.sparse import dok_matrix, coo_matrix
from random import sample
import random
random.seed(123)
from graphlab import SFrame
# import graphlab as gl
# from graphlab import SGraph
# import numpy as np
# from datetime import datetime
# from numpy.linalg import norm


class Transformer(object):

    def __init__(self):
        # read movies and tags
        self.movies = pd.read_csv("data/ml-20m/movies.csv", quotechar='"')
        self.tags = pd.read_csv("data/ml-20m/tags.csv", quotechar='"')
        self.ratings = pd.read_csv("data/ml-20m/ratings.csv", quotechar='"')

        # ratings.mtx
        # movieId => movieRow
        self.movie_dict = dict((m, i)
                               for (i, m) in enumerate(self.movies['movieId']))
        # userId => userRow
        self.user_dict = dict(
            (u, i) for (i, u) in enumerate(self.ratings['userId'].unique()))

        self.nm, self.nr, self.nu = len(self.movies), len(
            self.ratings), len(self.user_dict)

        self.genres = self.movies['genres'].map(lambda x: set(x.split('|')))
        ug = reduce(lambda a, b: a | b, self.genres, set())
        self.unique_genres = filter(lambda x: x != '(no genres listed)', ug)
        self.ng = len(self.unique_genres)

        remove_space = lambda s: ''.join(e for e in str(s) if e.isalnum())
        self.tags['tag'] = self.tags['tag'].apply(remove_space)

        # genreId => genreRow
        self.genres_dict = dict((t, i)
                                for (i, t) in enumerate(self.unique_genres))
        # tagId => tagRow
        self.tags_id_dict = dict(
            (t, i) for (i, t) in enumerate(self.tags['tag'].unique()))
        self.nt = len(self.tags_id_dict)

        self.movie_tags = self.tags.groupby(['movieId', 'tag']).agg({'userId': 'count'}) \
            .rename(columns={'userId': 'count'}).reset_index()

    def get_rating_matrix(self, samples=None, save_to=None):
        # pylint: disable=redefined-outer-name
        """
            get (and write) rating matrix from long format
        """

        if samples is None:
            samples = len(self.ratings)

        mat = dok_matrix((self.nu, self.nm))
        print save_to, 'samples length =', len(samples)

        for t, i in enumerate(samples):
            if t % 500000 == 0:
                print t
            uid = self.user_dict[self.ratings['userId'][i]]
            mid = self.movie_dict[self.ratings['movieId'][i]]
            rating = self.ratings['rating'][i]
            mat[uid, mid] = rating
        if save_to is not None:
            sio.mmwrite(save_to, mat)
            print "done writing", save_to
        return mat

    def get_rating_sf(self, samples, save_to=None):
        sf = SFrame(self.ratings.ix[samples])
        sf['userId'] = sf['userId'].apply(lambda uid: self.user_dict[uid])
        sf['movieId'] = sf['movieId'].apply(lambda mid: self.movie_dict[mid])
        if save_to is not None:
            print "saving sframe to", save_to
            sf.save(save_to)
        return sf

    def save_ratings_splits_mtx(self, train_ids, valid_ids, test_ids, name):
        if len(valid_ids) > 0:
            self.get_rating_matrix(
                valid_ids, save_to="data/%s_validate.mtx" % name)
        self.get_rating_matrix(test_ids, save_to="data/%s_test.mtx" % name)
        self.get_rating_matrix(train_ids, save_to="data/%s_train.mtx" % name)

    def save_ratings_splits_sf(self, train_ids, valid_ids, test_ids, name):
        if len(valid_ids) > 0:
            valid_sf = self.get_rating_sf(
                valid_ids, save_to="data/%s_validate.sframe" % name)
        test_sf = self.get_rating_sf(
            test_ids, save_to="data/%s_test.sframe" % name)
        train_sf = self.get_rating_sf(
            train_ids, save_to="data/%s_train.sframe" % name)
        return train_sf, valid_sf, test_sf

    # given list X, randomly split data into training, validation and test set

    def debug_split(self, S=None, split_ratio=20):
        if S is None:
            S = range(self.nr)
        # smaller split for debugging
        debug_size = self.nr / split_ratio
        debug_sample = sample(S, debug_size)
        train_ids, valid_ids, test_ids = self.sample_split(debug_size)

        get_debug = lambda ids: map(lambda i: debug_sample[i], ids)

        train_ids = get_debug(train_ids)
        valid_ids = get_debug(valid_ids)
        test_ids = get_debug(test_ids)

        return train_ids, valid_ids, test_ids

    def get_movie_matrix(self, nht, save_to=None):
        nm = self.nm
        ng = self.ng
        mat = dok_matrix((nm, ng + nht))

        for i, gs in enumerate(self.genres):
            for g in gs:
                if g in self.genres_dict:
                    mat[i, self.genres_dict[g]] = 1

        for i in xrange(len(self.movie_tags)):
            mid = self.movie_dict[self.movie_tags['movieId'][i]]
            t = self.movie_tags['tag'][i]
            # use hash kernel
            hh = hash(t)
            h, e = (hh / 2) % nht, -1 if hh % 2 == 0 else 1
            ti = ng + h
            mat[mid, ti] += e * (self.movie_tags['count'][i] ** 0.25)

        m = mat.tocsr()
        print 'dividing'

        sqrt2 = 2 ** 0.5
        for i in xrange(nm):
            sg = (m[i, :ng].data ** 2).sum() ** 0.5
            if sg > 0:
                m[i, :ng] /= sg * sqrt2
            st = (m[i, ng:].data ** 2).sum() ** 0.5
            if st > 0:
                m[i, ng:] /= st * sqrt2

            if i % 3000 == 0:
                print "div %d" % i, (m[i].data ** 2).sum()
        m.eliminate_zeros()
        if save_to is not None:
            sio.mmwrite(save_to, m)
        return m


def sample_split(n, n_test=None, n_validate=None):
    n_test = n / 5 if n_test is None else n_test
    n_validate = 0 if n_validate is None else n_validate

    samples = sample(range(n), n_test + n_validate)
    test_ids = samples[0:n_test]
    validate_ids = samples[n_test:]
    train_ids = list(set(range(n)) - set(test_ids) - set(validate_ids))

    return train_ids, validate_ids, test_ids


def gen_cs(ratings_matrix, r_test=0.2, r_validate=0, name='ratings_cs'):
    (_, nm) = ratings_matrix.shape
    n_test = int(r_test * nm)
    n_validate = int(r_validate * nm)

    ratings_matrix_csc = ratings_matrix.tocsc()

    samples = sample(range(nm), n_test + n_validate)
    test_movie_ids = samples[0:n_test]
    validate_movie_ids = samples[n_test:]
    train_movie_ids = list(set(range(nm)) - set(samples))

    train_ratings_mtx = csc_cols_to_zero(
        ratings_matrix_csc.copy(), test_movie_ids + validate_movie_ids)
    test_ratings_mtx = csc_cols_to_zero(
        ratings_matrix_csc.copy(), train_movie_ids + validate_movie_ids)
    if n_validate > 0:
        valid_ratings_tmx = csc_cols_to_zero(
            ratings_matrix_csc.copy(), train_movie_ids + test_movie_ids)
        sio.mmwrite('data/%s_validate.mtx' % name, valid_ratings_tmx)

    sio.mmwrite('data/%s_train.mtx' % name, train_ratings_mtx)
    sio.mmwrite('data/%s_test.mtx' % name, test_ratings_mtx)
    return train_ratings_mtx, valid_ratings_tmx, test_ratings_mtx


def get_debug(data):
    full_train = sio.mmread('data/%s_train.mtx' % data).tocsr()
    (nu, nm) = full_train.shape

    print 'sampling'
    debug_mids = sample(range(nm), nm / 5)
    debug_uids = sample(range(nu), nu / 5)

    debug = full_train[debug_uids][:, debug_mids].tocoo()
    nr = debug.nnz
    train_ids, _, test_ids = sample_split(nr)

    # build matrix from given indices
    print 'writing debug_train'
    debug_train = coo_matrix(
        (debug.data[train_ids], (debug.row[train_ids], debug.col[train_ids])), debug.shape)
    sio.mmwrite('data/%s_debug_train.mtx' % data, debug_train)
    print 'writing debug_test'
    debug_test = coo_matrix(
        (debug.data[test_ids], (debug.row[test_ids], debug.col[test_ids])), debug.shape)
    sio.mmwrite('data/%s_debug_test.mtx' % data, debug_test)

    # build movie mtx from debug_mids
    print 'movie debug'
    movies = sio.mmread('data/movies.mtx').tocsr()
    movies_debug = movies[debug_mids]
    sio.mmwrite('data/movies_%s_debug.mtx' % data, movies_debug)

    return debug, debug_train, debug_test, movies_debug


# get_debug('ratings')


def csc_col_to_zero(csc, col):
    csc.data[csc.indptr[col]: csc.indptr[col + 1]] = 0


def csc_cols_to_zero(csc, cols):
    for col in cols:
        csc_col_to_zero(csc, col)
    csc.eliminate_zeros()
    return csc


def gen_cs_ratio(ratings_matrix, r_test=0.2, r_test2=0.3, r_validate=0, name='ratings_cs'):
    (nu, nm) = ratings_matrix.shape
    n_test = int(r_test * nm)
    n_validate = int(r_validate * nm)

    ratings_matrix_csc = ratings_matrix.tocsc()

    samples = sample(range(nm), n_test + n_validate)
    test_movie_ids = samples[0:n_test]
    validate_movie_ids = samples[n_test:]
    train_movie_ids = list(set(range(nm)) - set(samples))

    test_user_ids = sample(range(nu), int(r_test2*nu))
    train_user_ids = list(set(range(nu)) - set(test_user_ids))

    train_ratings_mtx = ratings_matrix_csc.copy().todok()
    test_ratings_mtx = csc_cols_to_zero(
        ratings_matrix_csc.copy(), train_movie_ids + validate_movie_ids).todok()

    for i, mid in enumerate(test_movie_ids):
        if i % 1000 == 0:
            print "test_movie_id", i
        for uid in train_user_ids:
            test_ratings_mtx[uid, mid] = 0
        for uid in test_user_ids:
            train_ratings_mtx[uid, mid] = 0

    if n_validate > 0:
        valid_ratings_tmx = csc_cols_to_zero(
            ratings_matrix_csc.copy(), train_movie_ids + test_movie_ids)
        sio.mmwrite('data/%s_validate.mtx' % name, valid_ratings_tmx)

    sio.mmwrite('data/%s_train.mtx' % name, train_ratings_mtx)
    sio.mmwrite('data/%s_test.mtx' % name, test_ratings_mtx)
    return train_ratings_mtx, valid_ratings_tmx, test_ratings_mtx


def get_sf_from_coo(coo, save_to):
    sf = SFrame({'userId': coo.row, 'movieId': coo.col, 'rating': coo.data})
    if save_to is not None:
        print "saving sframe to", save_to
        sf.save(save_to)
    return sf


def normal_split(nr, T):
    train_ids, valid_ids, test_ids = sample_split(nr)
    T.save_ratings_splits_mtx(train_ids, valid_ids, test_ids, 'ratings_normal')
    T.save_ratings_splits_sf(train_ids, valid_ids, test_ids, 'ratings_normal')
    return train_ids, valid_ids, test_ids


#     save_ratinenres = movies['genres'].map(lambda x: set(x.split('|')))
#
# def two_round_split():
#     train_ids, _, test_ids = sample_split(nr)
#     debug_train_ids, _, debug_valid_ids = debug_split(train_ids)
#     save_ratings_splits_sf(ratings, debug_train_ids, [], debug_valid_ids, 'ratings_debug')
#     save_ratings_splits_sf(ratings, train_ids, [], test_ids, 'ratings')
#     save_ratings_splits_mtx(ratings, debug_train_ids, [], debug_valid_ids, 'ratings_debug')
#     save_ratings_splits_mtx(ratings, train_ids, [], test_ids, 'ratings')
#     return train_ids, test_ids, debug_train_ids, debug_valid_ids
