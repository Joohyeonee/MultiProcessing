# MultiProcessing
1. CPU 병렬처리
 - Python 라이브러리 사용 : multiprocessing의 Pool 모듈 이용
```
from multiprocessing import Pool

#병렬처리 작업을 함수화
def matmul(input_tuple):
  result = input_tuple[0] * input_tuple[1]
  return result

#병렬처리에 사용할 프로세스 개수 지정
pool = Pool(proceesses = 2)

#Pool 함수의 입력값으로 병렬처리할 함수와 입력값을 리스트로 전달
result = p.map(matmul, [a, b]) #input list a, b
print(result)
```
 - OpenCL 라이브러리 사용 : 플랫폼, 디바이스 인덱스 설정을 통해 CPU 코어로 병렬처리(4코어, 6코어 등등)
 > clGetPlatformIds()를 통해 플랫폼 개수(GPU, CPU)를 받아온 후 platform 공간 할당 => clGetPlatformIDs()로 플랫폼을 받아옴
 > clGetDeviceIDs()의 device_type을 CL_DEVICE_CPU로 설정
