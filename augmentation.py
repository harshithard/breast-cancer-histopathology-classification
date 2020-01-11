from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
import numpy as np
import os
from imutils import paths
import random
augmentedimagegenerator = ImageDataGenerator(rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
from google.colab import drive
drive.mount('/content/drive')
dataset_benign='data/benign'
imagepaths_benign=sorted(list(paths.list_images(dataset_benign)))
print(len(imagepaths_benign))
random.seed(42)
dataset_benign='data/benign'
imagepaths_benign=sorted(list(paths.list_images(dataset_benign)))
print(len(imagepaths_benign))
random.shuffle(imagepaths_benign)
augmented_dataset_benign='augmentedimages/benign'
augmented_imagepaths_benign=sorted(list(paths.list_images(augmented_dataset_benign)))
print(len(augmented_imagepaths_benign))
for imgpath in imagepaths_benign:
  try:
    image=load_img(imgpath)
    x = img_to_array(image) 
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in augmentedimagegenerator.flow(x, batch_size=64,save_to_dir='augmentedimages/benign', save_prefix='aug', save_format='png'):
      i=i+1
      if i>2:
        break
  except Exception as e:
    print(e)
  finally:
    if len(augmented_imagepaths_benign)>400:
      break
print(len(augmented_imagepaths_benign))

random.seed(42)
dataset_ductal_carcinoma='data/ductal_carcinoma'
imagepaths_ductal_carcinoma=sorted(list(paths.list_images(dataset_ductal_carcinoma)))
print(len(imagepaths_ductal_carcinoma))
random.shuffle(imagepaths_ductal_carcinoma)
augmented_dataset_ductal_carcinoma='augmentedimages/ductal_carcinoma'
augmented_imagepaths_ductal_carcinoma=sorted(list(paths.list_images(augmented_dataset_ductal_carcinoma)))
print(len(augmented_imagepaths_ductal_carcinoma))
for imgpath in imagepaths_ductal_carcinoma:
  try:
    image=load_img(imgpath)
    x = img_to_array(image) 
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in augmentedimagegenerator.flow(x, batch_size=64,save_to_dir='augmentedimages/ductal_carcinoma', save_prefix='aug', save_format='png'):
      i=i+1
      augmented_imagepaths_ductal_carcinoma=sorted(list(paths.list_images(augmented_dataset_ductal_carcinoma)))
      if i>2:
        break
  except Exception as e:
    print(e)
  finally:
    if len(augmented_imagepaths_ductal_carcinoma)>400:
      break
print(len(augmented_imagepaths_ductal_carcinoma))
random.seed(42)
dataset_ductal_carcinoma='data/ductal_carcinoma'
imagepaths_ductal_carcinoma=sorted(list(paths.list_images(dataset_ductal_carcinoma)))
print(len(imagepaths_ductal_carcinoma))
random.shuffle(imagepaths_ductal_carcinoma)
augmented_dataset_ductal_carcinoma='augmentedimages/ductal_carcinoma'
augmented_imagepaths_ductal_carcinoma=sorted(list(paths.list_images(augmented_dataset_ductal_carcinoma)))
print(len(augmented_imagepaths_ductal_carcinoma))
for imgpath in imagepaths_ductal_carcinoma:
try:
    image=load_img(imgpath)
    x = img_to_array(image) 
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in augmentedimagegenerator.flow(x, batch_size=64,save_to_dir='augmentedimages/ductal_carcinoma', save_prefix='aug', save_format='png'):
      i=i+1
      augmented_imagepaths_ductal_carcinoma=sorted(list(paths.list_images(augmented_dataset_ductal_carcinoma)))
      if i>2:
        break
  except Exception as e:
    print(e)
  finally:
    if len(augmented_imagepaths_ductal_carcinoma)>400:
      break
print(len(augmented_imagepaths_ductal_carcinoma))

random.seed(42)
dataset_lobular_carcinoma='data/lobular_carcinoma'
imagepaths_lobular_carcinoma=sorted(list(paths.list_images(dataset_lobular_carcinoma)))
print(len(imagepaths_lobular_carcinoma))
random.shuffle(imagepaths_lobular_carcinoma)
augmented_dataset_lobular_carcinoma='augmentedimages/lobular_carcinoma'
augmented_imagepaths_lobular_carcinoma=sorted(list(paths.list_images(augmented_dataset_lobular_carcinoma)))
print(len(augmented_imagepaths_lobular_carcinoma))
for imgpath in imagepaths_lobular_carcinoma:
  try:
    image=load_img(imgpath)
    x = img_to_array(image) 
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in augmentedimagegenerator.flow(x, batch_size=32,save_to_dir='augmentedimages/lobular_carcinoma', save_prefix='aug', save_format='png'):
      i=i+1
      augmented_imagepaths_lobular_carcinoma=sorted(list(paths.list_images(augmented_dataset_lobular_carcinoma)))
      if i>2:
        break
  except Exception as e:
    print(e)
  finally:
    if len(augmented_imagepaths_lobular_carcinoma)>1000:
      break
print(len(augmented_imagepaths_lobular_carcinoma))
 random.seed(42)
dataset_mucinous_carcinoma='data/mucinous_carcinoma'
imagepaths_mucinous_carcinoma=sorted(list(paths.list_images(dataset_mucinous_carcinoma)))
print(len(imagepaths_mucinous_carcinoma))
random.shuffle(imagepaths_mucinous_carcinoma)
augmented_dataset_mucinous_carcinoma='augmentedimages/mucinous_carcinoma'
augmented_imagepaths_mucinous_carcinoma=sorted(list(paths.list_images(augmented_dataset_mucinous_carcinoma)))
print(len(augmented_imagepaths_mucinous_carcinoma))
for imgpath in imagepaths_mucinous_carcinoma:
  try:
    image=load_img(imgpath)
    x = img_to_array(image) 
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in augmentedimagegenerator.flow(x, batch_size=32,save_to_dir='augmentedimages/mucinous_carcinoma', save_prefix='aug', save_format='png'):
      i=i+1
      augmented_imagepaths_mucinous_carcinoma=sorted(list(paths.list_images(augmented_dataset_mucinous_carcinoma)))
      if i>2:
        break
  except Exception as e:
    print(e)
  finally:
    if len(augmented_imagepaths_mucinous_carcinoma)>800:
      break
print(len(augmented_imagepaths_mucinous_carcinoma))
 random.seed(42)
dataset_papillary_carcinoma='data/papillary_carcinoma'
imagepaths_papillary_carcinoma=sorted(list(paths.list_images(dataset_papillary_carcinoma)))
print(len(imagepaths_papillary_carcinoma))
random.shuffle(imagepaths_papillary_carcinoma)
augmented_dataset_papillary_carcinoma='augmentedimages/papillary_carcinoma'
augmented_imagepaths_papillary_carcinoma=sorted(list(paths.list_images(augmented_dataset_papillary_carcinoma)))
print(len(augmented_imagepaths_papillary_carcinoma))
for imgpath in imagepaths_papillary_carcinoma:
  try:
    image=load_img(imgpath)
    x = img_to_array(image) 
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in augmentedimagegenerator.flow(x, batch_size=32,save_to_dir='augmentedimages/papillary_carcinoma', save_prefix='aug', save_format='png'):
      i=i+1
      augmented_imagepaths_papillary_carcinoma=sorted(list(paths.list_images(augmented_dataset_papillary_carcinoma)))
      if i>2:
        break
  except Exception as e:
    print(e)
  finally:
    if len(augmented_imagepaths_papillary_carcinoma)>1000:
      break
print(len(augmented_imagepaths_papillary_carcinoma))
