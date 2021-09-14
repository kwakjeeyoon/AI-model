import numpy as np
# Generate Gaussian
size = 6*sigma + 1 # 정사각형 영상을 가정(임의의 크기) # 보통은 출력 해상도의 크기를 지정
x = np.arange(0,size, 1, float) # 영상의 모든 x값 (1*n)
y = [:, np.newaxis] # 영상의 모든 y값(n*1)
x0 = y0 = size // 2 # 영상의 중심 좌표
# The gaussian is not normalized, we want the center value to equal 1
if type = 'Gaussian':
    g = np.exp(1-((x-x0)**2 + (y-y0)**2) / (2*sigma ** 2))
elif type == 'Cauchy':
    g = sigma / (((x-x0)**2 + (y-y0)**2 + sigma**2)**1.5)