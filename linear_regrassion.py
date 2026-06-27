import tensorflow as tf


# 키와 신발 사이즈는 어떤 관련이 있는지 딥러닝으로 찾아내어 키로 신발 사이즈 예측하는 간단한 프로그램

tall = 180
shose_size = 290

# 신발 = 키 * a + b

# 파라미터
a = tf.Variable(0.1)
b = tf.Variable(0.2)

# 손실함수  (실제값 - 예측값)^2
def loss_function():
    return tf.square(shose_size - (tall * a + b))


# 경사하강법 자동으로 해주는 함수 (optimizers 설정으로 러닝레이트도 설정 가능)
# opt = tf.keras.optimizers.Adam(learning_rate=0.1)
opt = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1)

for i in range(300):  # 경사하강 300번 반복
    opt.minimize(loss_function, var_list = [a, b])  # <- 경사하강 자동으로 한 번 실행

print(f"예측한 신발 사이즈: {tall * a.numpy() + b.numpy():.2f}")