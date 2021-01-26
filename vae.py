import numpy as np
import matplotlib.pyplot as plt  # 画图


from keras.layers import Input, Dense, Lambda  # 输入层 全连接层 计算用的
from keras.models import Model
from keras import backend as K  # 后端 函数
from keras import objectives
from keras.datasets import mnist  # 数据集


batch_size = 100  # 每一批的大小是100个，每次拿100个数据训练
original_dim = 784  # 原始的维度28*28
intermediate_dim = 256  # 第一个全连接层的输出维度
latent_dim = 2  # 隐变量取2维只是为了方便后面画图
# 两个全连接层 784->256 256->2
epochs = 50  # 50轮

x = Input(shape=(original_dim,))  # 逗号呢是数据数量不确定
h = Dense(intermediate_dim, activation='relu')(x)  # intermediate_dim输出维度
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
    return z_mean + K.exp(z_log_var / 2) * epsilon


z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# 784-256-2-256-784


def vae_loss(x, x_decoded_mean):
    xent_loss = original_dim*objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var -
                            K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss+kl_loss


vae = Model(x, x_decoded_mean)
vae.compile(optimizer='rmsprop', loss=vae_loss)


# 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

vae.fit(x_train, x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test))

# 构建encoder，然后观察各个数字在隐空间的分布
encoder = Model(x, z_mean)

x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()

# 构建生成器
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# 观察能否通过控制隐变量的均值来输出特定类别的数字
n = 20  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
grid_x = np.linspace(-4, -4, n)
grid_y = np.linspace(-4, -4, n)

for i, xi in enumerate(grid_x):
    for j, yi in enumerate(grid_y):
        z_sample = np.array([[yi, xi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[(n-i-1) * digit_size: (n-i) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.show()
