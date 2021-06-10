import numpy as np
from keras import applications
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.utils import np_utils
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Input
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import optimizers
import glob
import numpy as np
import pandas as pd
import os
import shutil 
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img


#EXTRACTING LABELS FROM FILENAMES

os.chdir('/data/mraap/splitDataset/test')

test = os.listdir()
print(len(test))
dir= os.listdir()
cwd = os.getcwd()
print(cwd)

### Function to extract labels fom filename in the given path ###
import numpy as np
print(len(dir))
def extract_labels(path, dir):
  
  """
    Extracts labels from the filename provided at the given path
    
    Arguments:
    path -- str type, path of the folder containing the files
    dir -- list, containing the names of the files
    
    Returns:
    labels -- labels extracted from the filenames, numpy array of shape (y, 1) containing 0 or 1 (0: notumor, 1: tumor)
    
    """
  
  path = str(path)
  list = path.split('/')

  
  if (list[-1] == 'test' or list[-1] == 'train'):
    labels = []
    for i in range(len(dir)):
      #if (dir[i].split('.')[0].split('-')[2] == 'True'):
      #if i [0:3] == 'Yes':
      if dir[i].startswith("Yes"):
        label = 1
      else: 
        label = 0
      labels.append(label)
    labels = np.array(labels)
    return labels
    
  else:
    print('Indefinite Call to the function')

y_validation = extract_labels(os.getcwd(), test)
print(y_validation.shape)
print(y_validation)

os.chdir('/data/mraap/splitDataset/train')

train = os.listdir()
y_train = extract_labels(os.getcwd(), train)
print(y_train.shape)
print(y_train)

#CONVERTING TRAIN AND TEST IMAGES INTO NUMPY ARRAYS


### Function to convert images in specified path into numpy arrays ###

from keras.preprocessing import image
def image_to_nparray(dir):
  """
    Converts image files from the given folder into numpy arrays
    
    Arguments:
    dir -- list, contains the filenames in given path (train or test folders)
    
    Returns:
    img_array -- numpy array of shape (m,h,w,c), contains the image values in form of np array
    
    """
  
  img_list = []
  for str in dir:
    img = image.load_img(str, target_size = (128, 128))
    x = image.img_to_array(img)
    img_list.append(x)
  img_array = np.array(img_list)
  del(img_list)
  return img_array

#Make sure we are in the train directory
os.getcwd()

x_train = image_to_nparray(train)
print(x_train.shape)

#Go to test directory
os.chdir('/data/mraap/splitDataset/test')

x_validation = image_to_nparray(test)
print(x_validation.shape)

#Go to train directory
os.chdir('/data/mraap/splitDataset/train')

#LOADING NUMPY ARRAYS FROM MEMORY
os.chdir('//data/mraap/splitDataset/train')

import numpy as np

train_imgs_scaled = x_train.astype('float32')
validation_imgs_scaled = y_train.astype('float32') 
train_imgs_scaled /= 255 
validation_imgs_scaled /= 255 


# encode text category labels 
from sklearn.preprocessing import LabelEncoder 
 
le = LabelEncoder() 
le.fit(y_train) 
train_labels_enc = le.transform(y_train) 
validation_labels_enc = le.transform(y_validation) 
 
#print([1495:1505], train_labels_enc[1495:1505])

train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, horizontal_flip=True, fill_mode='nearest')
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(x_train, train_labels_enc,batch_size=10)
val_generator = val_datagen.flow(x_validation, validation_labels_enc, batch_size=10)

#build VGG network base
input_tensor = Input(shape=(128,128,3))
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
print('Model loaded.')

#build a classifier model to put on top of the CNN
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))  
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

#These next two steps are to keep the model sequential
# copy all the layers of VGG16 to model
model = Sequential()
for l in base_model.layers:
    model.add(l)

# concatenate VGG16 and top model 
model.add(top_model)

for layer in model.layers[:15]:
    layer.trainable=False
    
#compile the model with SGD/momentum optimizer
#and a very slow learning rate
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])


history = model.fit_generator(train_generator, 
                              steps_per_epoch=200, 
                              epochs=50,
                              validation_data=val_generator, 
                              validation_steps=200, 
                              verbose=1)

model.save('/data/mraap/VGG16_FineTuningModel.h5')


from keras.models import load_model

scores_train = model.evaluate(train_generator, steps = 56)
print("Train Accuracy = ", scores_train[1])

scores = model.evaluate(val_generator, steps = 56)
print("Validation Accuracy =", scores[1])
print(history.history['accuracy'])
print(history.history['loss'])
print(history.history['val_accuracy'])
print(history.history['val_loss'])
