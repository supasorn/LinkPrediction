# 3-way split


from lib_transform import Transformer, sample_split  # , get_debug

T = Transformer()

train_ids, validate_ids, test_ids = sample_split(T.nr, n_validate=T.nr/5)
T.save_ratings_splits_mtx(train_ids, validate_ids, test_ids, 'ratings3normal')
T.save_ratings_splits_sf(train_ids, validate_ids, test_ids, 'ratings3normal')
