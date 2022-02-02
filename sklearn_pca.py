import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

gradeDF = pd.DataFrame({
  'name': ['멍개0', '멍개1', '멍개2', '멍개3', '멍개4','멍개5','멍개6','멍개7','멍개8', '멍개9'],
  'korean': [95, 90, 80, 60, 40, 80, 95, 30, 15, 60],
  'english': [95, 95, 75, 70, 35, 80, 90, 25, 10, 70]
})

print('원본 데이터 프레임: ', )
print(gradeDF)

names = gradeDF['name']
grades = gradeDF[['korean', 'english']]
scaler = StandardScaler()

grade_scaler = scaler.fit_transform(grades.values) # 데이터 정규화
print('데이터 정규화 shape: ',grade_scaler.shape)
print('데이터 정규화: ', )
print(grade_scaler)

pca = PCA()
pca.fit(grade_scaler)

# PCA의 결과는 결국 고유 벡터를 출력한다.
# 고유 벡터는 원본 데이터의 가장 큰 분산을 나타낼 수 있는 벡터가 된다.
# 해당 벡터를 이용하여 기존 데이터를 해당 벡터를 기준으로 다시 설정할 수 있다.
print('고유값(PCA 성분): ' ) 
print(pca.components_)

# 1. 기존 데이터에 고유벡터 방향 그리기
plt.subplot(1,2,1)
plt.plot(
  [grade_scaler[i][0] for i in range(len(grade_scaler))], 
  [grade_scaler[i][1] for i in range(len(grade_scaler))], 
  'ro'
)
plt.axis([-5, 5, -5, 5])
# 첫 번째 고유벡터 그리기
plt.quiver(0, 0, pca.components_[0][0], pca.components_[0][1], color = 'blue') # 고유값: 1.99

# 두 번째 고유벡터 그리기
plt.quiver(0, 0, pca.components_[1][0], pca.components_[1][1], color = 'green') # 고유값: 0.02
# plt.grid()
# plt.show()

# 2. 고유 벡터를 기반으로 원본 데이터 재설정
plt.subplot(1,2,2)
PC1 = [grade_scaler[i][0] * pca.components_[0][0] + grade_scaler[i][1] * pca.components_[0][1] for i in range(len(grade_scaler))]
PC2 = [grade_scaler[i][0] * pca.components_[1][0] + grade_scaler[i][1] * pca.components_[1][1] for i in range(len(grade_scaler))]

print('주성분(고유벡터)를 기준으로 만들어진 데이터: ', )
print(PC1)
print('주성분(고유벡터)를 기준으로 만들어진 데이터: ', )
print(PC2)

plt.plot(PC1, PC2, 'ro')

plt.axis([-5, 5, -5, 5])
plt.grid()
plt.show()