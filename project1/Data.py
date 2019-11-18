import numpy as np
import math
import matplotlib.pyplot as plt
import os.path

def Data(k,dim,n):
    if os.path.exists("data" + str(k) + ".npy"):
        data = np.load("data" + str(k) + ".npy")
        return data
    data = []
    radius = np.random.randint(10,100)
    rad = np.full(k,radius)
    rad = np.square(rad)
    sum = np.zeros(k)
    count= k
    for i in range(dim-1):
        data.append(np.random.rand(k))
        sum+=np.square(data[i])
    last_col = rad-sum
    last_col = np.sqrt(np.abs(last_col))
    data.append(last_col)
    while(count!=n):
        point = np.random.rand(k)
        radius_of_point = np.sum(np.square(point))
        if(math.sqrt(radius_of_point)<radius):
            data.append(point)
            count+=1
    data_write = np.array(data).T
    np.save("data" + str(k)+".out", data_write)
    return data_write
