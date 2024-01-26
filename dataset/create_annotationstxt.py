import cv2
import sys
import os
import csv
# import pandas as pd

# from PIL import Image
import os
import shutil
import random
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import StratifiedKFold



def split_txt():

    # Load your dataset from a text file
    with open('annotations_filtered_peak_all.txt', 'r') as file:
        data = file.readlines()

    # Assuming each line has a label at the end (e.g., 'data... class_id')
    labels = [line.split()[-1] for line in data]

    # Split the data into train, validation, and test sets with equal class distribution
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)
    val_data, test_data, val_labels, test_labels = train_test_split(test_data, test_labels, test_size=0.5, random_state=42, stratify=test_labels)

    # Define file names for the splits
    train_file = 'train.txt'
    val_file = 'val.txt'
    test_file = 'test.txt'

    # Write the data to separate files
    with open(train_file, 'w') as file:
        file.writelines(train_data)

    with open(val_file, 'w') as file:
        file.writelines(val_data)

    with open(test_file, 'w') as file:
        file.writelines(test_data)

    print(f'Dataset has been split and saved into {train_file}, {val_file}, and {test_file}.')

# split_txt()


# create five folds with the given  file and name the text files as train_fold1.txt, test_fold1.txt train_fold2.txt, test_fold2.txt and so on

def createfolds(txt_file_path):
    
    # Step 1: Load the Data
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()

    data = []
    labels = []

    for line in lines:
        label = line.strip()[-1]  # Assuming label is the last character
        text = line.strip()[:-1]  # Assuming text is everything except the last character
        labels.append(label)
        data.append(text)

    # Step 2: Stratified Split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold split

    for fold, (train_idx, test_idx) in enumerate(skf.split(data, labels)):
        train_data = [data[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]

        test_data = [data[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]

        # Step 3: Save Folds
        with open(f'fold_{fold+1}_train.txt', 'w') as train_file:
            for label, text in zip(train_labels, train_data):
                train_file.write(f'{text}{label}\n')

        with open(f'fold_{fold+1}_test.txt', 'w') as test_file:
            for label, text in zip(test_labels, test_data):
                test_file.write(f'{text}{label}\n')


createfolds('annotations_filtered_peak_2.txt')

def create_annotxt():
    bl = 'BL'
    one= 'PA1'
    two= 'PA2'
    three= 'PA3'
    four= 'PA4'

    root_dir = '/home/livia/work/Biovid/PartB/biovid_classes'

    #loop to go through all the subdirectories in the source directory

    file = open("annotations_filtered_peak_all.txt", "w")

    for sub_dir in os.listdir(root_dir):
        if sub_dir.endswith('.txt'):
            continue    
        sub_dir_path = os.path.join(root_dir, sub_dir)
        videos_list = os.listdir(sub_dir_path)
        for file_dir in videos_list:
            sub_video_path = os.path.join(sub_dir_path, file_dir)
            #count nyumber of images in each folder
            count = 0
            for sub_video in os.listdir(sub_video_path):
                count = count + 1
            
            if count == 75: 
                if sub_dir == '0' or sub_dir == '1' or sub_dir == '2' or sub_dir == '3' or sub_dir == '4':    
                    # write_file = os.path.join(sub_dir, file_dir) + " " + '1' + " " +str(count) + " " + str(sub_dir)
                    if sub_dir == '0':
                        class_label = '0'
                    elif sub_dir == '1':
                        class_label = '1'
                    elif sub_dir == '2':
                        class_label = '2'
                    elif sub_dir == '3':
                        class_label = '3'
                    elif sub_dir == '4':
                        class_label = '4'
                    write_file = os.path.join(sub_dir, file_dir) + " " + '50' + " " + "70" + " " + class_label # only peak
                    file.write(write_file + "\n")
    file.close()    








# def count_labels(file_path):
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#         labels = [line.strip().split()[-1] for line in lines]

#     label_counts = {}
#     for label in labels:
#         if label in label_counts:
#             label_counts[label] += 1
#         else:
#             label_counts[label] = 1

#     return label_counts

# # File paths for train, validation, and test sets
# train_file = 'train.txt'
# val_file = 'val.txt'
# test_file = 'test.txt'

# # Count labels in each file
# train_label_counts = count_labels(train_file)
# val_label_counts = count_labels(val_file)
# test_label_counts = count_labels(test_file)

# # Print label counts
# print(f"Label counts in train set: {train_label_counts}")
# print(f"Label counts in validation set: {val_label_counts}")
# print(f"Label counts in test set: {test_label_counts}")



# code to write current file name and folder name to a text file
# import os
# import sys
#
# # Open a file
# path = "/var/www/html/"
# dirs = os.listdir( path )
#   
# # This would print all the files and directories
# for file in dirs:
#    print (file)
#    f = open("demofile2.txt", "a")
#    f.write(file + "\n")
#    f.close()


