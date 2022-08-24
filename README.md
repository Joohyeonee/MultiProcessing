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
 > * Platform : OpenCL 병렬처리가 가능한 환경(CPU, GPU)
 > * Device : 연산 가능한 유닛(코어)
 > * Queue : 호스트(버퍼를 생성해서 보내고 연산된 결과 버퍼를 받는 곳)에서 디바이스(연산을 수행하는 곳) 별로 생성하는 queue
 > * Program :  빌드된 바이너리(kernel)
 > * Kernel : GPU에서 구동되는 함수
 > * clGetPlatformIDs()를 통해 플랫폼 개수(GPU, CPU)를 받아온 후 platform 공간 할당 
 > * clGetDeviceIDs()의 device_type을 CL_DEVICE_CPU로 설정(gpu : CL_DEVICE_GPU)
