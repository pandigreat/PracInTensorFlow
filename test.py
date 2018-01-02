from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import struct

dic = [[1,2,2,3]]
dic.append([0 for i in range(4)])
print(dic)
dic = np.array(dic)
print(dic)
print(np.column_stack(dic, np.zeros(1, 4)))