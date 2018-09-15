from keras.layers import Input, Dense, Flatten, Reshape
from keras.models import Model, Sequential
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D
from keras.callbacks import Callback
from keras.datasets import mnist
import numpy as np
import wandb
from wandb.keras import WandbCallback
from util import Images


def add_noise(x_train, x_test):
    noise_factor = 1.0
    x_train_noisy = x_train + np.random.normal(loc=0.0, scale=noise_factor, size=x_train.shape) 
    x_test_noisy = x_test + np.random.normal(loc=0.0, scale=noise_factor, size=x_test.shape) 
    
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)
    return x_train_noisy, x_test_noisy

run = wandb.init()
config = run.config

config.encoding_dim = 32
config.epochs = 10

(x_train, _), (x_test, _) = mnist.load_data()
(x_train_noisy, x_test_noisy) = add_noise(x_train, x_test)


x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

img_w = 28;
img_h = 28;

model = Sequential()

# add conv
model.add(Reshape((img_w,img_h,1), input_shape=(img_w,img_h)))

model.add(Conv2D(32, (3, 3),  activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3),  activation='relu', padding='same'))

model.add(Flatten())
model.add(Dense(config.encoding_dim, activation='relu'))
model.add(Dense(img_w*img_w, activation='sigmoid'))
model.add(Reshape((img_w,img_h)))
model.compile(optimizer='adam', loss='binary_crossentropy')


model.fit(x_train_noisy, x_train,
                epochs=config.epochs,
                validation_data=(x_test_noisy, x_test), 
                callbacks=[Images(), WandbCallback(save_model=False)])


model.save("auto-denoise.h5")




