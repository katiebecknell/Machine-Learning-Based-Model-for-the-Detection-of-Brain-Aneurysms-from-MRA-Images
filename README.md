# Machine Learning Based Model for the Detection of Brain Aneurysms from MRA Images

Our project aims to detect the presence of brain aneurysms from MRA images

How to run our code:

1. Ensure that all data images are in the form of .jpg and are two folders: Positive (containing only images that have aneurysms present) and Negative (containing only images that do not have aneuyrsms present)
2. After the images are renamed, move the files into one singluar dataset
      a. First, create a directory called dataset
                mkdir dataset/
      b. Move the positive image files to the newly created dataset
                python POSITIVE_file_move.py
      c. Move the negative image files to the newly created dataset
                python NEGATIVE_file_move.py
