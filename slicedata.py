
# coding: utf-8

# In[1]:

from __future__ import division
import scipy as sp
import numpy as np
from scipy import io


# In[2]:

f = io.mmread("data/netflix_mm")


# In[3]:

f.shape


# In[6]:

io.mmwrite("data/netflix_mm_50000_5000", f.tocsc()[:,:5000].tocsr()[:50000,:])


# In[ ]:



