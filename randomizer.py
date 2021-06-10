import os
import shutil
import glob
import random
import math

files_glob = glob.glob("/data/mraap/dataset/*_*_*-*-*.jpg")
print(files_glob)
num_files = len(files_glob)

print("\n TOTAL FILES: ", num_files)

train_num = math.floor(0.34 * num_files)
print("\n # OF TRAIN IMAGES: ", train_num)

validation_num = math.floor(0.34 * num_files)
print("\n # OF VALIDATION IMAGES: ", validation_num)



train_to_be_moved = random.sample(glob.glob("/data/mraap/dataset/*_*_*-*-*.jpg"),train_num)
 
for f in enumerate(train_to_be_moved, 1):
    dest = os.path.join("/data/mraap/splitDataset/train")
    if not os.path.exists(dest):
        os.makedirs(dest)
    shutil.move(f[1], dest)

validation_to_be_moved = random.sample(glob.glob("/data/mraap/dataset/*_*_*-*-*.jpg"), validation_num)


for i in enumerate(validation_to_be_moved, 1):
    dest = os.path.join("/data/mraap/splitDataset/validation/")
    if not os.path.exists(dest):
        os.makedirs(dest)
    shutil.move(i[1], dest)

test_to_be_moved = (glob.glob("/data/mraap/dataset/*_*_*-*-*.jpg"))
print("\n # OF TEST IMAGES: ", len(test_to_be_moved))

for i in enumerate(test_to_be_moved, 1):
    dest = os.path.join("/data/mraap/splitDataset/test")
    if not os.path.exists(dest):
        os.makedirs(dest)
    shutil.move(i[1], dest)
