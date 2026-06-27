import tensorflow as tf


# 기본 텐서 만들기 (그냥 숫자, 리스트 같은 거 담는 자료형)
tensor = tf.constant([3, 4, 5])
tensor2 = tf.constant([6, 7, 8])

print(tensor + tensor2)
print(tf.add(tensor, tensor2))

# 행렬 연산을 머신러닝에서 많이 쓰게 되는데 그때 텐서라는 자료형이 행렬과 비슷한 역할을 한다
# (행렬로 인풋/w값 저장가능 -> 그럼 노드값 계산식이 쉬워짐)

# 텐서연산 함수 - tf.add(), tf.subtract(), tf.divide(), tf.multiply(), tf.matmul() - 행렬의 곱

tensor3 = tf.constant([ [1, 2], 
                        [3, 4]])

tensor4 = tf.zeros(10)  # 0이 10개로 차있는 텐서 생성
tensor5 = tf.zeros( [2,2] )   # 2행 2열로 0으로 가득찬 텐서 생성
#  [2,2,3] 3개의 0이 들어있는 걸 2개 만들고 그걸 2개 만들어 달라는 뜻 (뒤에서부터 해석)

# tensor의 shape
print(tensor3.shape)

# weight 저장하고 싶으면 Variable만들기
w = tf.Variable(1.0)

print(w)

# 고정된 값을 저장할 땐 constant, 변동을 쉽게 하려면 Variable 
# (weghit는 Variable로 작성)

w.assign(2)  # 변수에 새로운 값 할당
print(w.numpy())  # 변수 저장된 값을 쓰거나 불러올 수 있음

# 변수에도 행렬 저장 가능