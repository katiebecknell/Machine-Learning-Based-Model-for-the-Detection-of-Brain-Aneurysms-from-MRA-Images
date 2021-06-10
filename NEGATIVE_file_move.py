import os
import shutil


root_directory_path = "/data/mraap/Negative/"
for root, dirname, filenames in os.walk(root_directory_path):
    for file in dirname:
        root_folder_path = root + file
        for count, filename in enumerate(os.listdir(root_folder_path)):
            
            #print("\ndestination path: " + new_name_destination_path)
            
            current_path = root_directory_path + file + "/" + filename
            
            #print("\ncurrent path: "  + original_image_path)
            
            move_image_path = "/data/mraap/dataset/" 
            
            shutil.copy(current_path, move_image_path) 
        
