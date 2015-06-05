import scipy.io as sio
from lib_transform import gen_cs

ratings_train = sio.mmread('data/ratings_train.mtx')
# ratings_validate = sio.mmread('data/ratings_validate.mtx')
ratings_test = sio.mmread('data/ratings_test.mtx')

ratings_matrix = ratings_train + ratings_test
gen_cs(ratings_matrix, 0.2, 0.2, 'full_ratings_cs')
