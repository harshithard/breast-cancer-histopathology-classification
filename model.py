from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.core import Activation,Flatten,Dense,Dropout
from keras import backend as k

from keras.preprocessing.image import ImageDataGenerator,img_to_array
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import os
import cv2
import pickle
import random
from sklearn.metrics import confusion_matrix

#to save images to background
#matplotlib.use("Agg")
dataset_benign='C:/Users/benign'
dataset_m_ductal='C:/Users/ductal_carcinoma'
dataset_m_lobular='C:/Users/lobular_carcinoma'
dataset_m_mucinous='C:/Users/mucinous_carcinoma'
dataset_m_papillary='C:/Users/papillary_carcinoma'
model_path="model.h5"
label_path="/"
plot_path="/"
HP_LR=1e-3
HP_EPOCHS=100
HP_BS=55
HP_IMAGE_DIM=(96,96,3)
data=[]
classes=[]
imagepaths_benign=sorted(list(paths.list_images(dataset_benign)))
imagepaths_ductal=sorted(list(paths.list_images(dataset_m_ductal)))
imagepaths_lobular=sorted(list(paths.list_images(dataset_m_lobular)))
imagepaths_mucinous=sorted(list(paths.list_images(dataset_m_mucinous)))
imagepaths_papillary=sorted(list(paths.list_images(dataset_m_papillary)))
print(len(imagepaths_benign))
print(len(imagepaths_ductal))
print(len(imagepaths_lobular))
print(len(imagepaths_papillary))
print(len(imagepaths_mucinous))
random.seed(42)
imagepaths=imagepaths_ductal+imagepaths_lobular+imagepaths_mucinous+imagepaths_papillary+imagepaths_benign
print(len(imagepaths))
random.shuffle(imagepaths)
print()
#print(imagepaths[1])
print(len(imagepaths))
classes=[]
for imgpath in imagepaths:
    try:
        image=cv2.imread(imgpath)
        image=cv2.resize(image,(96,96))
        image_array=img_to_array(image)
        data.append(image_array)
        label=imgpath.split('/')[-1]
        temp=label.split(os.path.sep)[-2]
        print(temp)
        #print(label)
        #if label in ['ductal_carcinoma','papillary_carcinoma','lobular_carcinoma','mucinous_carcinoma']:
        #  label='malignant'
        #print(label)
        classes.append(temp)    
    except Exception as e:
        print(e)
print(classes)
#normalization
data=np.array(data,dtype=float)
data=data/255.0
labels=np.array(classes)
lb=LabelBinarizer()
labels=lb.fit_transform(labels)
print(len(data))
print(classes[0])
print(labels[0])
print(classes[6])
print(labels[6])
print(len(data))
print(len(labels))
xtrain,xtest,ytrain,ytest=train_test_split(data,labels,test_size=0.2,random_state=42)
aug=ImageDataGenerator(rotation_range=0.25,width_shift_range=0.25,height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
#model=tinyVGG.build(height=96,width=96,depth=3,classes=len(lb.classes_))
model=Sequential()
input_shape=(96,96,3)
channel_dim=-1

#large patterns in smaller images
model.add(Conv2D(32,(3,3),padding='same',input_shape=input_shape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(MaxPooling2D(pool_size=(3,3)))
#model.add(Dropout(0.25))
#increase filter and reduce pool size to find better and finer patterns
model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),padding='same'))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(Conv2D(128,(3,3),padding='same'))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.25))

"""
#Deleted layers 

model.add(Conv2D(128,(3,3),padding='same'))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(Conv2D(128,(3,3),padding='same'))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),padding='same'))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(Conv2D(128,(3,3),padding='same'))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(256,(3,3),padding='same'))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(Conv2D(256,(3,3),padding='same'))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

"""
model.add(Flatten())#converts into a single dimensional array

model.add(Dense(1024))
model.add(Activation('relu'))

model.add(BatchNormalization())
 
model.add(Dropout(0.5))

model.add(Dense(5))
model.add(Activation("softmax"))

print(model.summary())
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
mc = ModelCheckpoint('weights.best.keras', monitor='val_acc', save_best_only=True)
aug=ImageDataGenerator(rotation_range=0.25,width_shift_range=0.25,height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
opt=Adam(lr=HP_LR,decay=HP_LR/HP_EPOCHS)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=['accuracy'])
history=model.fit_generator(aug.flow(xtrain,ytrain,batch_size=HP_BS),validation_data=(xtest,ytest),steps_per_epoch=len(xtrain)//HP_BS,epochs=HP_EPOCHS)
model.save('mymodcat2.h5')

#loss = history.history['loss']
#accuracy = history.history['acc']
gt=[]
pred=[]
predictions=model.predict(xtest)

for i in range(len(xtest)):
    pred.append(np.argmax(predictions[i]))
   
print(pred)
for i in range(len(xtest)):
    for j in range(5):
        if ytest[i][j]==1:
            gt.append(j)
           
           
     
for i in range(100):
    print(np.argmax(predictions[i])+1)
    print(ytest[i])
print(gt)

print(len(xtest))
count=0
for i in range(len(xtest)):
    if pred[i]!=gt[i]:
        count+=1
print(count)
print(confusion_matrix(pred,gt))
print("testing accuracy=",1-(count/len(xtest)))
x=[]
y=[]
for i in range(len(predictions)):
    for j in range(5):
        if ytrain[i][j]==1:
             x.append(j)
for i in range(len(predictions)):
    y.append(np.argmax(predictions[i])+1)
   
   
         
plt.plot(y,x)
plt.show()

import matplotlib.pyplot as plt
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()