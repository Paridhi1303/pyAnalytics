#Topic:Correlation, Covariance,
#-----------------------------
#libraries

#The difference between variance, covariance, and correlation is:

#Variance is a measure of variability from the mean
#Covariance is a measure of relationship between the variability of 2 variables - covariance is scale dependent because it is not standardized
#Correlation is a of relationship between the variability of of 2 variables - correlation is standardized making it not scale dependent


import pandas as pd
import numpy as np


# Setting a seed so the example is reproducible
np.random.seed(4272018)
df = pd.DataFrame(np.random.randint(low= 0, high= 20, size= (5, 2)),  columns= ['Commercials Watched', 'Product Purchases'])#creating a data frame of size 5*2, integer values range from 0 to 20
df
df.agg(["mean", "std"])
df.cov()
df.corr()
#correlation value is 0.107077

#skewness & Kurtosis
#positive skewness, mean is shifted to the right, negative skew data means mean is shifted to the left, but graph shifted to right
#%matplotlib inline
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from scipy.stats import skew

import matplotlib.pyplot as plt

plt.style.use('ggplot')

data = np.random.normal(0, 1, 10000000) #doubt
data
len(data)
np.var(data)

plt.hist(data, bins=60)

print("mean : ", np.mean(data))#mean of normally distributed data is almost zero
print("var  : ", np.var(data))#square of std deviation which is one for normally distributed data
print("skew : ",skew(data))#almost zero
print("kurt : ",kurtosis(data))
#https://www.spcforexcel.com/knowledge/basic-statistics/are-skewness-and-kurtosis-useful-statistics
#https://www.researchgate.net/post/What_is_the_acceptable_range_of_skewness_and_kurtosis_for_normal_distribution_of_data distribution

#If skewness is between -0.5 to 0.5, data is fairly symmetrical, if skewness is between 0.5 to 1 or -1 to -0.5, data is moderately skewed, if skewness is <-1 or >1, data is highly skewed


import numpy as np
from scipy.stats import kurtosis, skew

x_random = np.random.normal(0, 2, 10000)

x = np.linspace( -5, 5, 10000 )
y = 1./(np.sqrt(2.*np.pi)) * np.exp( -.5*(x)**2  )  # normal distribution

np.skewness(y)
skew(y)

import matplotlib.pyplot as plt

f, (ax1, ax2) = plt.subplots(1, 2)
ax1.hist(x_random, bins='auto')
ax1.set_title('probability density (random)')
ax2.hist(y, bins='auto')
ax2.set_title('(your dataset)')
plt.tight_layout()
