import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
k = 50
points = np.random.rand(k, 2)   # 30 random points in 2-D

hull = ConvexHull(points,incremental = True)
v = hull.vertices
print(len(v))
while( len(v)!= k):
    rand_point=np.random.rand(1, 2)
    points= np.append(points,rand_point,axis=0)
    np.savetxt('data2.out', points, delimiter=',')
    hull.add_points(rand_point,restart=True)
    v=hull.vertices
    print(len(v))

plt.plot(points[:,0], points[:,1], 'o')
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

plt.show()