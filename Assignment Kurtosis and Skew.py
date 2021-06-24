# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 18:35:23 2021

@author: rukap
"""

#find skewness and kurtosis of mtcars data, state whether they are left or right skewed
import pandas as pd
import numpy as np
import numpy as np
from scipy.stats import kurtosis, skew
from pydataset import data
import matplotlib.pyplot as plt

plt.style.use('ggplot')
mtcars=data('mtcars')
data=mtcars
data
data.columns
skew(data, axis=0)
kurtosis(mtcars.hp)

data.mean(axis=0)
data.median(axis=0)
mtcars.hp.quantile(q=0.5)
skew(mtcars.mpg)
plt.hist(mtcars.mpg, bins=5)
#Thus it is positively skewed and moderately skewed

kurtosis(mtcars.mpg)
#negative kurtosis

#Find which columns are normally distibuted, skewed and kurtosis

plt.hist(mtcars.mpg, bins=60)
