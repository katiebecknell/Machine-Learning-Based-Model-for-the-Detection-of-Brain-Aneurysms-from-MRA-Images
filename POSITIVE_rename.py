import os
import shutil


root_directory_path = "/data/mraap/Positive/"
for root, dirname, filenames in os.walk(root_directory_path):
    for file in dirname:
        root_folder_path = root + file
        for count, filename in enumerate(os.listdir(root_folder_path)):

            new_image_name = "Yes_" + file + "_" + filename

            #print("new file name: " + new_image_name)

            original_image_path = root_folder_path + "/" + filename

            new_name_destination_path = root_directory_path + file + "/" + new_image_name

            os.rename(original_image_path, new_name_destination_path)
