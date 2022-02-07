# https://blog.naver.com/pjt3591oo/222641548110

from PIL import Image
from sklearn.decomposition import PCA
from numpy import asarray
import os

ORIGIN_IMAGE_PATH = 'test.png'
COMPRESION_IMAGE_PATH = 'compresion.png'

# Open the image form working directory
origin_image = Image.open(ORIGIN_IMAGE_PATH)

# RGBA 를 분리한다
splited_img = Image.Image.split(origin_image) # R(red), G(green), B(blue), A(opacity)
red_img = asarray(splited_img[0])
green_img = asarray(splited_img[1])
blue_img = asarray(splited_img[2])
opacity_img = asarray(splited_img[3])

# 분리된 R, G, B, A 벡터마다 적용할 PCA 모델생성
pca_r = PCA(n_components=50)
pca_g = PCA(n_components=50)
pca_b = PCA(n_components=50)
pca_a = PCA(n_components=50)

pca_r.fit(red_img)
trans_pca_r = pca_r.transform(red_img)
pca_g.fit(green_img)
trans_pca_g = pca_g.transform(green_img)
pca_b.fit(blue_img)
trans_pca_b = pca_b.transform(blue_img)
pca_a.fit(opacity_img)
trans_pca_a = pca_a.transform(opacity_img)

print('compression')
print(trans_pca_r.shape, trans_pca_g.shape, trans_pca_b.shape, trans_pca_a.shape)

# 축소된 차원 복구
r_arr = pca_r.inverse_transform(trans_pca_r)
g_arr = pca_g.inverse_transform(trans_pca_g)
b_arr = pca_b.inverse_transform(trans_pca_b)
a_arr = pca_b.inverse_transform(trans_pca_a)

print('recovery')
print(r_arr.shape, g_arr.shape, b_arr.shape, a_arr.shape)

# 복구된 R, G, B 행렬 데이터 하나의 이미지로 복구
recoveryImage = Image.merge('RGBA', (
  Image.fromarray(r_arr).convert('L'), 
  Image.fromarray(g_arr).convert('L'),
  Image.fromarray(b_arr).convert('L'),
  Image.fromarray(a_arr).convert('L')
))

# 저장
recoveryImage.save(COMPRESION_IMAGE_PATH)

print()
print('원본 데이터 width * height')
print(origin_image.size)
print('압축 데이터 width * height')
print(recoveryImage.size)

print()
print(recoveryImage)

p1_size = os.path.getsize(ORIGIN_IMAGE_PATH)
p2_size = os.path.getsize(COMPRESION_IMAGE_PATH)

print('원본 데이터 크기', p1_size)
print('압축 데이터 크기', p2_size)
