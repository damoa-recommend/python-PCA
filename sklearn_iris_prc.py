import ssl

from sklearn.metrics import explained_variance_score
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, fetch_olivetti_faces
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA, SparsePCA
from sklearn.preprocessing import StandardScaler

from plot import *

def plot_iris2(iris, labels, plt):
  plt.figure()
  colors = ['navy', 'purple', 'red']

  for xy, label in zip(iris, labels):
    plt.scatter(xy[0], xy[1], c=colors[label])

def plot_iris1(iris, labels, plt):
  plt.figure()
  colors = ['navy', 'purple', 'red']

  for xy, label in zip(iris, labels):
    plt.scatter(xy[0], 0, c=colors[label])

iris, labels = load_iris(return_X_y=True)
iris = StandardScaler().fit_transform(iris) # 데이터 표준화

print()
print('iris dataset: ')
print(iris[0:3])
# print('iris label: ')
# print(labels[0: 3])

print('before feature')
print(iris.shape) 

pca = PCA(n_components=2) # 주성분 수를 2로 지정
pca.fit(iris) # 고유벡터, 고유값을 찾는다.
# 고유벡터인 PC(고유벡터)를 기준으로 데이터를 다시 반영(회전)한다.(component가 2인 경우 PC를 x, y축으로 변경(회전))
transformed_iris = pca.transform(iris) 

print('before feature')
print(transformed_iris.shape)

print('고유값(eigen value)')
print(pca.explained_variance_ratio_) # 고유값

print('고유벡터(eigen vector)')
print(pca.components_) # 고유벡터

print('새로운 값')
print(transformed_iris[: 4])

print('분산 설명량')
explained_variance_score = sum(pca.explained_variance_ratio_)
print(explained_variance_score)

print()
print('값 검증(원본 데이터를 통해 고유벡터(주성분)으로 새로 만든 데이터 transformed_iris가 iris와 주성분 결과인 pca.components_와 계산(내적) 했을 때 정말 같은가?')
print('해당 검산은 데이터 표준화를 진행해야 한다.')
print(iris[0], pca.components_[0])
print(iris[0].dot(pca.components_[0]), transformed_iris[0][0])

# 만약 컴포넌트가 1개라면? 
# plt.scatter(xy[0], xy[1], c=colors[label]) y축을 0으로 해야한다.
plot_iris2(iris, labels, plt) 
plt.title('original data')

plot_iris2(transformed_iris, labels, plt) 

plt.title('principal components: 2EA, variance: %f'%(explained_variance_score))

pca = PCA(n_components=1)
pca.fit(iris)
transformed_iris = pca.transform(iris)
explained_variance_score = sum(pca.explained_variance_ratio_)
plot_iris1(transformed_iris, labels, plt) 
plt.title('principal components: 1EA, variance: %f'%explained_variance_score)

plt.show()