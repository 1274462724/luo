# %%
from keras.preprocessing.image import ImageDataGenerator,img_to_array
from keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16 
from keras import backend as K 
import numpy as np 
from keras.models import Sequential
print(__doc__)

# %%
# 通道处理
img_width,img_height =224,224
if K.image_data_format() == "channels_first":
    input_shape = (3,img_width,img_height)
else:
    input_shape = (img_width,img_height,3)

# %%
# VGG16网络结构
def VGGNet_16():
    model = Sequential()
    # 第一块64
    #       1
    model.add(Conv2D(64,(3,3),input_shape=input_shape,activation='relu',padding='same'))
    #       2
    model.add(Conv2D(64,(3,3),activation ='relu',padding='same'))
    #    池化
    model.add(MaxPool2D((2,2),2))
    # 第二块128
    #       1
    model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
    #       2
    model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
    #       池化
    model.add(MaxPool2D((2,2),2))
    # 第三块区域256
    #       1
    model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
    #       2
    model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
    #       3
    model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
    #       池化
    model.add(MaxPool2D((2,2),2))
    # 第四块区域512
    #       1
    model.add(Conv2D(512,(3,3),activation='relu',padding='same'))  
    #       2
    model.add(Conv2D(512,(3,3),activation='relu',padding='same'))  
    #       3
    model.add(Conv2D(512,(3,3),activation='relu',padding='same'))
    #       池化
    model.add(MaxPool2D((2,2),2))
    # 第五块区域 512
    #       1
    model.add(Conv2D(512,(3,3),activation='relu',padding='same'))
    #       2
    model.add(Conv2D(512,(3,3),activation='relu',padding='same'))
    #       3
    model.add(Conv2D(512,(3,3),activation='relu',padding='same'))
    #       池化
    model.add(MaxPool2D((2,2),2))
    #       拉平
    model.add(Flatten())

    # 第六块区域全连接 4096
    #       1
    model.add(Dense(4096,activation='relu'))
    #       最优失活参数0.5
    model.add(Dropout(0.5))
    #       2
    model.add(Dense(4096,activation='relu'))
    #       最优失活参数0.5
    model.add(Dropout(0.5))
    #       输出层1000,但是猫狗分类只有2
    model.add(Dense(1,activation='sigmoid'))
    
    return model

# %%
# 查看VGG模型架构
model = VGGNet_16()
model.summary()
# %%
# 模型编译
model.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])
# %%
# 数据批量预处理
train_data_dir = r'D:\Python_project\Jupyter_project\Four_week\dogs_vs_cats_datas\train'
validation_data_dir = r'D:\Python_project\Jupyter_project\Four_week\dogs_vs_cats_datas\validation'
nb_train_samples = 1083
nb_validation_samples = 400
epochs  =1
batch_size = 20
train_datagen = ImageDataGenerator(rescale=1.0/255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1.0/255)

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
# 模型训练
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)
# %%
