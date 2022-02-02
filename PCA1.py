import numpy as np
import math
import matplotlib.pyplot as plt
from analysis import covariance_matrix

# PC1고 PC2로 변경(PCA0.py에서 구한 PCA값)
data1 = [-1.49, -1.42, -0.71, 0.0, 1.35, -0.78, -1.42, 1.92, 2.63, 0.0]
data2 = [0.05, -0.16, 0.13, -0.28, 0.09, 0.06, 0.12, 0.1, 0.11, -0.28]

print('첫 번째 데이터: ', data1)
print('두 번째 데이터: ', data2)

cov_mat = covariance_matrix(data1, data2)
print('공분산 행렬:', cov_mat)

mat = np.array(cov_mat)

w, v = np.linalg.eig(mat)
print('고유값', np.round(w, 2))
print('고유백터', np.round(v, 2)) 

plt.plot(data1, data2, 'ro')

plt.axis([-5, 5, -5, 5])

plt.quiver(0, 0, v[0][0], v[0][1], color = 'blue') # 첫 번쨰 고유벡터 그리기
plt.quiver(0, 0, v[1][0], v[1][1], color = 'green')  # 두 번쨰 고유벡터 그리기
plt.grid()
plt.show()