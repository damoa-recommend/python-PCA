import numpy as np
import math

# 공분산 행렬
def covariance_matrix(data1, data2):
  data1_average = sum(data1)/len(data1)
  data2_avergrage = sum(data2)/len(data2)

  data1_deviation = [a - data1_average for a in data1] # 편차
  data2_deviation = [a - data2_avergrage for a in data2] # 편차

  data1_variance = [(a - data1_average) ** 2 for a in data1] # 분산 - 표준편차 1 - 음수를 없애기 위해 편차 제곱
  data2_variance = [(a - data2_avergrage) ** 2 for a in data2] # 분산 - 표준편차 1 - 음수를 없애기 위해 편차 제곱

  data1_variance = sum(data1_variance)/len(data1_variance) # 분산 - 표준편차 2 - 평균을 구하여 분산을 구한다
  data2_variance = sum(data2_variance)/len(data2_variance) # 분산 - 표준편차 2 - 평균을 구하여 분산을 구한다

  data1_std_deviation = math.sqrt(float(data1_variance)) # 표준편차 3 - 제곱을 했기 떄문에 제곱근을 하여 표준편차를 구한다
  data2_str_deviation = math.sqrt(float(data2_variance)) # 표준편차 3 - 제곱을 했기 떄문에 제곱근을 하여 표준편차를 구한다

  data_len = len(data1_deviation) if len(data1_deviation) < len(data2_deviation) else len(data2_deviation)

  cov = [data1_deviation[idx] * data2_deviation[idx] for idx in range(0, data_len)]
  cov = sum(cov) / len(cov)
  relation = cov / (data1_std_deviation * data2_str_deviation)

  print('첫 번째 데이터 분산: %s'%(np.round(data1_variance, 2)))
  print('두 번째 데이터 분산: %s'%(np.round(data2_variance, 2)))

  print('첫 번째 데이터 표준편차: %s'%(math.sqrt(np.round(data1_std_deviation, 2))))
  print('두 번째 데이터 표준편차: %s'%(math.sqrt(np.round(data2_str_deviation, 2))))

  print('공분산: %s'%(np.round(cov, 2)))
  print('상관계수: %s'%(relation))
  return [
    [np.round(data1_variance, 2), np.round(cov, 2)], # 분산, 공분산
    [np.round(cov, 2), np.round(data2_variance, 2)] # 공분산, 분산
  ]

# 표준화
def scale_less(data):
  return np.round((data - np.mean(data, axis=0)) / np.std(data, axis=0), 1)
