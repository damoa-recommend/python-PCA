import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

print('고유값(PCA 성분): ' ) # PCA의 결과는 결국 고유 벡터를 출력한다.
print(pca.components_)