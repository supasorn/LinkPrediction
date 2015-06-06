from scipy.io import sio
from lib_transform import get_sf_from_coo

for r in ['', '_cs']:
    for d in ['_debug', '']:
        for t in ['_train', '_test']:
            filename = 'ratings' + r + d + t
            print filename
            coo = sio.mmread('data/%s.mtx' % filename)
            get_sf_from_coo(coo, 'data/%s.sgraph' % filename)
