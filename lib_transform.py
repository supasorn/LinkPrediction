import scipy.io as sio
import pandas as pd
from scipy.sparse import dok_matrix
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

        # save_ratings_splits(ratings, train_ids, valid_ids, test_ids, 'ratings_debug')
        # save_ratings_splits_mtx(ratings, train_ids, valid_ids, test_ids, 'ratings_debug')

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
            if i % 10000 == 0:
                print "div %d" % i
            sg = (m[i, :ng].data ** 2).sum() ** 0.5
            if sg > 0:
                m[i, :ng].data /= sg * sqrt2
            st = (m[i, ng:].data ** 2).sum() ** 0.5
            if st > 0:
                m[i, ng:].data /= st * sqrt2
        m.eliminate_zeros()
        return m


def sample_split(n, n_test=None, n_validate=None):
    n_test = n / 5 if n_test is None else n_test
    n_validate = 0 if n_validate is None else n_validate

    samples = sample(range(n), n_test + n_validate)
    test_ids = samples[0:n_test]
    validate_ids = samples[n_test:]
    train_ids = list(set(range(n)) - set(test_ids) - set(validate_ids))

    return train_ids, validate_ids, test_ids


def csc_col_to_zero(csc, col):
    csc.data[csc.indptr[col]: csc.indptr[col + 1]] = 0


def csc_cols_to_zero(csc, cols):
    for col in cols:
        csc_col_to_zero(csc, col)
    csc.eliminate_zeros()
