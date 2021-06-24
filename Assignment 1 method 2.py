# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 20:15:55 2021

@author: rukap
"""

from pydataset import data
mtcars = data('mtcars')
dataDF = mtcars
dataDF.skew()
dataDF.kurtosis()
