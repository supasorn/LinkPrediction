
# coding: utf-8

# Assume the data has been loaded from 
# http://files.grouplens.org/datasets/movielens/ml-20m.zip and unpacked in data/ml-20m

# In[1]:

import scipy.io as sio
import pandas as pd
from scipy.sparse import dok_matrix
from random import sample
import random
random.seed(123)
import graphlab as gl
from graphlab import SFrame
from graphlab import SGraph
import numpy as np


# In[2]:

#read movies and tags
movies = pd.read_csv("data/ml-20m/movies.csv", quotechar='"')
tags = pd.read_csv("data/ml-20m/tags.csv", quotechar='"')
ratings = pd.read_csv("data/ml-20m/ratings.csv", quotechar='"')


# In[3]:

### ratings.mtx
# movieId => movieRow
movie_dict = dict((m, i) for (i, m) in enumerate(movies['movieId']))
# userId => userRow
user_dict = dict((u, i) for (i, u) in enumerate(ratings['userId'].unique()))


# In[4]:

nm, nr, nu = len(movies), len(ratings), len(user_dict)


# In[5]:

# nu, nm


# In[6]:

def get_rating_matrix(ratings, samples, save_to=None):
    """
        get (and write) rating matrix from long format
    """
    mat = dok_matrix((nu, nm))
    print save_to, 'samples length =', len(samples)

    for t, i in enumerate(samples):
        if t % 500000 == 0: print t
        uid = user_dict[ratings['userId'][i]]
        mid = movie_dict[ratings['movieId'][i]]
        rating = ratings['rating'][i]
        mat[uid,mid] = rating
    if save_to is not None:
        sio.mmwrite(save_to, mat)
        print "done writing", save_to
    return mat


# In[7]:

# def get_rating_matrix_from_sf(sf, save_to=None):
#     """
#         get (and write) rating matrix from long format 
#     """
#     mat = dok_matrix((nu, nm))
#     n = len(sf)
#     print 'save %s  samples length = %s' %(save_to, n)

#     for i in xrange(n):
#         if i % 10000 == 0: print i
#         r = sf[i] 
#         u, m = r['userId'], r['movieId']
#         mat[u,m] = r['rating']
#     if save_to is not None:
#         sio.mmwrite(save_to, mat)
#         print "done writing", save_to
#     return mat


# In[8]:

def get_rating_sf(ratings, samples, save_to=None):
    sf = SFrame(ratings.ix[samples])
    sf['userId'] = sf['userId'].apply(lambda uid: user_dict[uid])
    sf['movieId'] = sf['movieId'].apply(lambda mid: movie_dict[mid])
    if save_to is not None:
        print "saving sframe to", save_to
        sf.save(save_to)
    return sf


# In[9]:

def save_ratings_splits_mtx(ratings, train_ids, valid_ids, test_ids, name):
    if len(valid_ids) > 0:
        get_rating_matrix(ratings, valid_ids, save_to="data/%s_validate.mtx"%name)
    get_rating_matrix(ratings, test_ids, save_to="data/%s_test.mtx"%name)
    get_rating_matrix(ratings, train_ids, save_to="data/%s_train.mtx"%name)


# In[10]:

def save_ratings_splits_sf(ratings, train_ids, valid_ids, test_ids, name):
    if len(valid_ids) > 0:
        valid_sf = get_rating_sf(ratings, valid_ids, save_to="data/%s_validate.sframe"%name)
    test_sf = get_rating_sf(ratings, test_ids, save_to="data/%s_test.sframe"%name)
    train_sf = get_rating_sf(ratings, train_ids, save_to="data/%s_train.sframe"%name)
    


# In[11]:

# given list X, randomly split data into training, validation and test set
def sample_split(n, n_test=None, n_validate=None):
    n_test = n/5 if n_test is None else n_test
    n_validate = 0 if n_validate is None else n_validate

    samples = sample(range(n), n_test + n_validate)
    test_ids = samples[0:n_test]
    validate_ids = samples[n_test:]
    train_ids = list(set(range(n)) - set(test_ids) - set(validate_ids))

    return train_ids, validate_ids, test_ids


# In[12]:

def debug_split(S=range(nr), split_ratio=20):
    # smaller split for debugging
    debug_size = nr/split_ratio
    debug_sample = sample(S, debug_size)
    train_ids, valid_ids, test_ids = sample_split(debug_size)
    
    get_debug = lambda ids: map(lambda i: debug_sample[i], ids)
    
    train_ids = get_debug(train_ids)
    valid_ids = get_debug(valid_ids)
    test_ids = get_debug(test_ids)

#     save_ratings_splits(ratings,  train_ids, valid_ids, test_ids, 'ratings_debug')
#     save_ratings_splits_mtx(ratings, train_ids, valid_ids, test_ids, 'ratings_debug')
    
    return train_ids, valid_ids, test_ids


# In[80]:

debug_train_ids, debug_valid_ids, debug_test_ids = debug_split(split_ratio=2000)
save_ratings_splits_sf(ratings, debug_train_ids, [], debug_test_ids, 'ratings_debug_small')


# In[ ]:

debug_train_ids, debug_valid_ids, debug_test_ids = debug_split()


# In[ ]:

# def normal_split():
#     train_ids, valid_ids, test_ids = sample_split(nr)
#     save_ratings_splits(ratings, train_ids, valid_ids, test_ids, 'ratings')
# #     save_ratinenres = movies['genres'].map(lambda x: set(x.split('|')))


# In[15]:

genres = movies['genres'].map(lambda x: set(x.split('|')))
unique_genres = reduce(lambda a,b: a|b, genres, set())
unique_genres = filter(lambda x: x!='(no genres listed)', unique_genres)
ng = len(unique_genres)

remove_space = lambda s: ''.join(e for e in str(s) if e.isalnum())
tags['tag'] = tags['tag'].apply(remove_space)

# genreId => genreRow
genres_dict = dict((t, i) for (i, t) in enumerate(unique_genres))
# tagId => tagRow
tags_id_dict = dict((t, i) for (i, t) in enumerate(tags['tag'].unique()))
nt = len(tags_id_dict)

print ng, nt


# In[16]:

unique_genres = reduce(lambda a,b: a|b, genres, set())
unique_genres = filter(lambda x: x!='(no genres listed)', unique_genres)
ng = len(unique_genres)

remove_space = lambda s: ''.join(e for e in str(s) if e.isalnum())
tags['tag'] = tags['tag'].apply(remove_space)

# genreId => genreRow
genres_dict = dict((t, i) for (i, t) in enumerate(unique_genres))
# tagId => tagRow
tags_id_dict = dict((t, i) for (i, t) in enumerate(tags['tag'].unique()))
nt = len(tags_id_dict)


# In[13]:

def two_round_split():
    train_ids, _, test_ids = sample_split(nr)
    debug_train_ids, _, debug_valid_ids = debug_split(train_ids)
    save_ratings_splits_sf(ratings, debug_train_ids, [], debug_valid_ids, 'ratings_debug')
    save_ratings_splits_sf(ratings, train_ids, [], test_ids, 'ratings')
    save_ratings_splits_mtx(ratings, debug_train_ids, [], debug_valid_ids, 'ratings_debug')
    save_ratings_splits_mtx(ratings, train_ids, [], test_ids, 'ratings')
    return train_ids, test_ids, debug_train_ids, debug_valid_ids


# In[16]:

train_ids, test_ids, debug_train_ids, debug_valid_ids = two_round_split()


# In[ ]:

# train_ids, valid_ids, test_ids = normal_split()


# In[46]:

movie_tags = tags.groupby(['movieId','tag']).agg({'userId':'count'})     .rename(columns={'userId': 'count'}).reset_index()


# In[57]:

def get_movie_matrix(nht, save_to=None):
    mat = dok_matrix((nm, ng + nht))

    for i, gs in enumerate(genres):
        for g in gs:
            if g in genres_dict:
                mat[i,genres_dict[g]] = 1

    for i in xrange(len(movie_tags)):
        mid = movie_dict[movie_tags['movieId'][i]]
        t = movie_tags['tag'][i]
        # use hash kernel
        hh = hash(t)
        h, e = (hh / 2) % nht , -1 if hh % 2 == 0 else 1
        ti = ng + h
        mat[mid, ti] += e * (movie_tags['count'][i] ** 0.25)
    
    m = mat.tocsr()
    for i in xrange(nm):
        rowmax = max(m[i, ng:].max(), -m[i, ng:].min())
        if rowmax > 0:
            m[i, ng:] /= rowmax 
    m.eliminate_zeros()
    
    if save_to is not None:
        sio.mmwrite(save_to, m)
    return m


# In[58]:

# write phi(m) -- features for each movie!
start= datetime.now()
movies_mtx = get_movie_matrix(40, 'data/movies.mtx')
print datetime.now()-start


# In[54]:

m = sio.mmread('data/movies.mtx').tocsr()
m.eliminate_zeros()
m


# In[ ]:

rating_matrix = get_rating_matrix(ratings, range(len(ratings)))


# In[52]:

test_movie_ids = set(sample(range(nm), nm/5))
train_movie_ids = set(range(nm)) - test_movie_ids


# In[58]:

def csc_col_to_zero(csc, col):
    csc.data[csc.indptr[col]: csc.indptr[col+1]] = 0


# In[59]:

def csc_cols_to_zero(csc, cols):
    for col in cols:
        csc_col_to_zero(csc, col)
    csc.eliminate_zeros()


# In[60]:

rating_matrix_csc = rating_matrix.tocsc()
train_rating_mtx = rating_matrix_csc.copy()
test_rating_mtx = rating_matrix_csc.copy()
csc_cols_to_zero(train_rating_mtx, test_movie_ids)
csc_cols_to_zero(test_rating_mtx, train_movie_ids)


# In[67]:

sio.mmwrite('data/ratings_cs_train.mtx', train_rating_mtx)
sio.mmwrite('data/ratings_cs_test.mtx', test_rating_mtx)


# In[68]:

def get_sf_from_coo(coo, save_to):
    sf = SFrame({'userId': coo.row, 'movieId': coo.col, 'rating': coo.data})
    if save_to is not None:
        print "saving sframe to", save_to
        sf.save(save_to)
    return sf


# In[69]:

train_rating_coo = train_rating_mtx.tocoo()
test_rating_coo = test_rating_mtx.tocoo()
get_sf_from_coo(train_rating_coo, 'data/ratings_cs_train.sgraph')
get_sf_from_coo(test_rating_coo, 'data/ratings_cs_test.sgraph')

