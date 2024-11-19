# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 16:39:53 2024

@author: cameron schmidt
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 100, 10)
y = np.arange(0,100, 10)

plt.stem(x,y)

plt.show()