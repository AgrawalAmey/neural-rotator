
# coding: utf-8

# In[1]:

import os
import sys

from keras import backend as K
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.layers import Add, BatchNormalization, Conv2D, Dense, Input, Lambda, MaxPooling2D, Reshape, UpSampling2D
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import rotate

# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

# from rot_coder.shortcut import shortcut

# In[2]:


# adapt this if using `channels_first` image data format
input_img = Input(shape=(28, 28, 1))
shape_input = Input(shape=(12,))
shape_input_reshaped = Reshape((1, 1, 12))(shape_input)

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(12, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(12, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(12, (3, 3), activation='relu', padding='same')(x)

encoded = MaxPooling2D((2, 2), padding='same')(x)

# Add + Upsample
x = Conv2D(12, (3, 3), activation='relu', padding='same')(encoded)
shape_vec = Lambda(lambda x: K.tile(x, [1, 2, 2, 1]))(shape_input_reshaped)
x = Add()([x, shape_vec])
x = UpSampling2D((2, 2))(x)

# Add + BN
x = Conv2D(12, (3, 3), activation='relu', padding='same')(x)
shape_vec = Lambda(lambda x: K.tile(x, [1, 4, 4, 1]))(shape_input_reshaped)
x = Add()([x, shape_vec])
x = BatchNormalization()(x)

# Add + Upsample
x = Conv2D(12, (3, 3), activation='relu', padding='same')(x)
shape_vec = Lambda(lambda x: K.tile(x, [1, 4, 4, 1]))(shape_input_reshaped)
x = Add()([x, shape_vec])
x = UpSampling2D((2, 2))(x)

# Add + BN
x = Conv2D(12, (3, 3), activation='relu', padding='same')(x)
shape_vec = Lambda(lambda x: K.tile(x, [1, 8, 8, 1]))(shape_input_reshaped)
x = Add()([x, shape_vec])
x = BatchNormalization()(x)

# Add + Upsample
x = Conv2D(12, (3, 3), activation='relu', padding='same')(x)
shape_vec = Lambda(lambda x: K.tile(x, [1, 8, 8, 1]))(shape_input_reshaped)
x = Add()([x, shape_vec])
x = UpSampling2D((2, 2))(x)


# Add + BN
x = Conv2D(12, (3, 3), activation='relu', padding='same')(x)
shape_vec = Lambda(lambda x: K.tile(x, [1, 16, 16, 1]))(shape_input_reshaped)
x = Add()([x, shape_vec])
x = BatchNormalization()(x)

# Upsample
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='last_conv')(x)

autoencoder = Model(inputs=[input_img, shape_input], outputs=decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# In[3]:


(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
# adapt this if using `channels_first` image data format
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
# adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

x_train_repeated = []
x_train_transformed = []
train_shapes = []
for i in range(x_train.shape[0]):
    for _ in range(5):
        rotation = np.random.randint(0, 12)
        x_train_repeated.append(x_train[i])
        x_train_transformed.append(rotate(x_train[i].reshape((28, 28)),
                            rotation * 36).reshape((28, 28, -1)))
        train_shape = np.zeros(12)
        train_shape[rotation] = 1
        train_shapes.append(train_shape)

del x_train

x_train_repeated = np.asarray(x_train_repeated)
x_train_transformed = np.asarray(x_train_transformed)
train_shapes = np.asarray(train_shapes)

x_test_transformed = []
test_shapes = []
for i in range(x_test.shape[0]):
    rotation = np.random.randint(0, 12)
    x_test_transformed.append(rotate(x_test[i].reshape((28, 28)),
                        rotation * 36).reshape((28, 28, -1)))
    test_shape = np.zeros(12)
    test_shape[rotation] = 1
    test_shapes.append(test_shape)

x_test_transformed = np.asarray(x_test_transformed)
test_shapes = np.asarray(test_shapes)


# In[ ]:
autoencoder.fit([x_train_repeated, train_shapes], x_train_transformed,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=([x_test, test_shapes], x_test_transformed),
                callbacks=[TensorBoard(log_dir='./logs/4')])


# In[ ]:
decoded_imgs = autoencoder.predict([x_test, test_shapes])

n = 5
plt.figure(figsize=(15, 10))
start_index = np.random.randint(1, 10000)
for i in range(start_index, start_index + n):
    # display original
    plot_index = i - start_index + 1
    ax = plt.subplot(3, n, plot_index)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display rotated
    plot_index = i + n - start_index + 1
    ax = plt.subplot(3, n, plot_index)
    plt.title(test_shapes[i].argmax() * 36)
    plt.imshow(x_test_transformed[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    plot_index = i - start_index + 1 + 2 * n
    ax = plt.subplot(3, n, plot_index)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


plt.show()
