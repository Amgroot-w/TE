# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 00:40:42 2018

@author: Arevalo
"""

import numpy as np
import pandas as pd

df = pd.read_table("d01.dat")#, sep="\s+", usecols=['TIME', 'XGSM'])
# size of a file df.shape