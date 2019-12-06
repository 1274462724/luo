# %%
import cv2 as cv
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Activation, Dropout, Add, Input, BatchNormalization, merge
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array
from keras import backend as K
import numpy as np
from keras.models import Model

import tensorflow as tf
# %%
img_wight, img_height = 50, 50
if K.image_data_format() == "channels_first":
    input_shape = (3, img_wight, img_height)
    bn_axis = 1
else:
    input_shape = (img_wight, img_height, 3)
    bn_axis = 3

input_img = Input(shape=input_shape)


# ResNet Module

x = Conv2D(64, (1, 1), padding='same',
           kernel_initializer='he_normal')(input_img)

x = BatchNormalization(axis=bn_axis)(x)
x = Activation('relu')(x)

x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
x = BatchNormalization(axis=bn_axis)(x)
x = Activation('relu')(x)

x = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal')(x)
x = BatchNormalization(axis=bn_axis)(x)

shortcut = Conv2D(256, (1, 1), padding='same',
                  kernel_initializer='he_normal')(input_img)
shortcut = BatchNormalization(axis=bn_axis)(shortcut)

x = Add()([x, shortcut])
x = Activation('relu')(x)

# fully connected layer
out = Flatten()(x)
out = Dense(48, activation='relu')(out)

# output layer
out = Dense(1, activation='sigmoid')(out)

model = Model(input_img, out)

model.summary()
# %%
# ????
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# %%
# %%
# ??ImageDataGenerator
train_data_dir = r'D:\Python_project\Jupyter_project\Four_week\dogs_vs_cats_datas\train'
validation_data_dir = r'D:\Python_project\Jupyter_project\Four_week\dogs_vs_cats_datas\validation'
nb_train_samples = 10800
nb_validation_samples = 4000
epochs = 1
batch_size = 20

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# %%
# ????
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

# %%
# ????
img = cv.resize(cv.imread(r'D:\Python_project\Jupyter_project\Four_week\dogs_vs_cats_datas\test\1.jpg'),
                (img_width, img_height)).astype(np.float32)

x = img_to_array(img)
x = np.expand_dims(x, axis=0)

relust = model.predict(x)

if relust[0] <= 0:
    print('?')
else:
    print('?')
