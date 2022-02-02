import numpy as np
import math
import matplotlib.pyplot as plt

from analysis import covariance_matrix, scale_less

korean = scale_less([95, 90, 80, 60, 40, 80, 95, 30, 15, 60])
english = scale_less([95, 95, 75, 70, 35, 80, 90, 25, 10, 70])

data_length = len(korean)

print('국어성적 z-score', korean)
print('영어성적 z-score', english)

cov_mat = covariance_matrix(korean, english)
print('공분산 행렬:', cov_mat)

mat = np.array(cov_mat)

w, v = np.linalg.eig(mat)
print('고유값', np.round(w, 2))
print('고유백터', np.round(v, 2)) 

# 고유벡터에 대응하는 값 구하기
PC1s = [np.round(korean[i] * v[0][0] + english[i] * v[0][1], 2) for i in range(data_length)]
PC2s = [np.round(korean[i] * v[1][0] + english[i] * v[1][1], 2) for i in range(data_length)]

print('새롭게 만들어 진 데이터 PCA')
print("PC1:", PC1s)
print("PC2:", PC2s)

plt.plot(korean, english, 'ro')
plt.axis([-5, 5, -5, 5])

# 첫 번째 고유벡터 그리기
plt.quiver(0, 0, v[0][0], v[0][1], color = 'blue') # 고유값: 1.99

# 두 번째 고유벡터 그리기
plt.quiver(0, 0, v[1][0], v[1][1], color = 'green') # 고유값: 0.02

plt.grid()
plt.show()