import glob
import numpy as np
import pandas as pd
import os
import shutil 
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

'''
#LOCATING DATA

#Need to change this from fetching data from google drive to data from our remote server
import os

os.chdir("/data/mraap/splitDataset")   # directory contains all the pictures sent as it is

files = os.listdir("/data/mraap/splitDataset")
print(len(files))       # 2125 images sent via the zip

#CREATING AND SPLITTING TRAIN AND TEST SETS

#Creating folder named "train" to store training set
train = 'train'

# Create target Directory if don't exist
if not os.path.exists(train):
  os.mkdir(train)
  print("Directory " , train ,  " Created ")
else:    
  print("Directory " , train ,  " already exists")

#Creating folder named "test" to store test set
test = 'test'

# Create target Directory if don't exist
if not os.path.exists(test):
  os.mkdir(test)
  print("Directory " , test ,  " Created ")
else:    
  print("Directory " , test ,  " already exists")

#Splitting data into "train" folder
import random
import shutil
dir= os.listdir()
cwd = os.getcwd()
print(cwd)
dir.sort()  # making sure that the filenames have a fixed order before shuffling
random.seed(23)  # to maintain the result as it is no matter how many times the code is run
random.shuffle(dir)  # shuffles the ordering of filenames

for file in files[:296]:  
  src = cwd + '/' + file
  dest = cwd + '/' + 'train/'  + file
  shutil.move(src, dest) 
  
# train ~ 85% of 2125
# test ~ 15% of 2125

#Checking the size of the train file
os.chdir("/data/mraap/dataset/train")
files = os.listdir()
print(len(files)) 

os.chdir("/data/mraap/dataset")

#NEED TO WRITE CODE TO PLACE REMAINING FILES IN THE TEST FOLDER
#The following commented code throws an error
#dir= os.listdir()
#cwd = os.getcwd()
#print(cwd)
#for file in files:  
  #src = cwd + '/' + file
  #dest = cwd + '/' + 'test/' + file
  #shutil.move(src, dest) 
  
'''

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

# Saving np label arrays in disk for future use and fast loading

#np.save("y_train.npy", y_train)      
#np.save("y_test.npy", y_test)  

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

# saving array to load it faster in future

#np.save("x_train.npy", x_train)      
#np.save("x_test.npy", x_test) 

#LOADING NUMPY ARRAYS FROM MEMORY
os.chdir('//data/mraap/splitDataset/train')

import numpy as np

#x_train = np.load('x_train.npy')
#x_test = np.load('x_test.npy')
#y_train = np.load('y_train.npy')
#y_test = np.load('y_test.npy')

train_imgs_scaled = x_train.astype('float32')
validation_imgs_scaled = y_train.astype('float32') 
train_imgs_scaled /= 255 
validation_imgs_scaled /= 255 
 
# visualize a sample image 
# print(train_imgs[0].shape) 
# array_to_img(train_imgs[0])

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

from keras.applications.resnet50 import ResNet50
from keras.models import Model
import keras


restnet = ResNet50(include_top=False, weights = 'imagenet', input_shape=(128,128,3))


output = restnet.layers[-1].output
output = keras.layers.Flatten()(output)
restnet = Model(restnet.input, outputs=output)


restnet.summary()


from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers

model = Sequential()
model.add(restnet)
model.add(Dense(512, activation='relu', input_dim=(128, 128, 3)))
model.add(Dropout(0.4))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])

for layer in model.layers[:]:
    print("layer: ", layer.name)
    #if (layer.name.find("conv5") != -1):
       # layer.trainable = True


model.summary()
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
  #                                      mode ="min", patience = 5, 
 #                                       restore_best_weights = True)
history = model.fit_generator(train_generator, 
                              steps_per_epoch=116, 
                              epochs=50, 
                              validation_data=val_generator,  
                              validation_steps = 116, 
                              verbose=1)

model.save('/data/mraap/featureExtraction.h5')

model = keras.models.load_model('/data/mraap/featureExtraction.h5')


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Train vs Validation Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
plt.savefig('/data/mraap/accuracyplotTransferLearning.jpg')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Train vs Validation Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
plt.savefig('/data/mraap/lossplotTransferLearning.jpg')

print(history.history['accuracy'])
print(history.history['loss'])
print(history.history['val_accuracy'])
print(history.history['val_loss'])

from keras.models import load_model

scores_train = model.evaluate(train_generator, steps = 10)
print("Train Accuracy = ", scores_train[1])

scores = model.evaluate(val_generator, steps = 10)
print("Validation Accuracy =", scores[1])
'''
print("WITH NEW WEIGHTS")
loaded_model = model.load_weights('cls_resnet_0.1.h5')
loss, acc = loaded_model.evaluate(train_generator, steps = 10)
print("Train Accuracy = ", acc)
loss2, acc2 = loaded_model.evaluate(val_generator, steps = 10)
print("Val Accuracy = ", acc2)

'''
