# Machine Learning Based Model for the Detection of Brain Aneurysms from MRA Images

Our project aims to detect the presence of brain aneurysms from MRA images

How to run our code:

1. Ensure that all data images are in the form of .jpg and are two folders: Positive (containing only images that have aneurysms present) and Negative (containing only images that do not have aneuyrsms present)
2. After the images are renamed, move the files into one singluar dataset
- First, create a directory called dataset:   mkdir dataset/
- Move the positive image files to the newly created dataset:   python POSITIVE_file_move.py
- Move the negative image files to the newly created dataset:   python NEGATIVE_file_move.py
3. Next, randomize all the image files in the dataset folder:   python randomizer.py. This will place the images into a new folder called splitDataset/.
4. Now the dataset is ready to be fed to the model. Depending on which model you wish to run use the following:     python ResNet50_FeatureExtraction.py
5. If you wish to run another model, firstly you will need to delete the dataset/ and splitDataset/ folders: rm -r dataset and rm -r splitDataset. After deleting these folders, repeat steps 1 - 4.
