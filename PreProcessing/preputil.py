import os
from pathlib import Path
from tqdm.notebook import tqdm
import pandas as pd
import json
import cv2
import numpy as np
from glob import glob

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ==============================
# Square Image
# ==============================
def perfect_square(img):
    x = np.shape(img)
    
    diff = np.abs(x[0] - x[1])
    a = int(diff/2)
    
    if(len(x) == 3):
        if (x[0] < x[1]):
            b = int(x[1]-(diff/2))
            res = img[:,a:b,:]
        else:
            b = int(x[0]-(diff/2))
            res = img[a:b,:,:]
    else:
        if (x[0] < x[1]):
            b = int(x[1]-(diff/2))
            res = img[:,a:b]
        else:
            b = int(x[0]-(diff/2))
            res = img[a:b,:]
        
    return res



# ==============================
# Resize Image
# ==============================
def resize_128(img):
    return cv2.resize(img,(128,128), interpolation=cv2.INTER_AREA)

def replace_labels(labels, labelfile, ignore_problems=False):
    labels = np.array(labels, dtype=object)
    mapped_labels = {}
    labelnames = pd.read_csv(labelfile, names=["id", "labelname"]).values
    if not ignore_problems:
        if len(labelnames) != len(np.unique(labelnames[:,0])):
            indices = np.unique(labelnames[:,0])
            raise Exception("Corrupt label names file. Some ids are associated with several labels:" + "".join(["\n\t" + str(i) + " " + str(labelnames[labelnames[:,0] == i,1]) for i in indices if np.count_nonzero(labelnames[:,0] == i) > 1]))
        if len(labelnames) != len(np.unique(labelnames[:,1])):
            indices = np.unique(labelnames[:,1])
            raise Exception("Corrupt label names file. Some label names occur more than once:" + "".join(["\n\t" + str(i) + " " + str(labelnames[labelnames[:,1] == i,0]) for i in indices if np.count_nonzero(labelnames[:,1] == i) > 1]))
    for row in labelnames:
        key = str(row[0]).strip()
        val = str(row[1]).strip()
        mapped_labels[key] = val
        indices = np.where(labels == key)[0]
        if len(indices) == 0:
            print(f"Warning: No instances for label {key} ({val})")
        else:
            labels[indices] = val
    unreplaced_labels = {l for l in labels if l not in list(mapped_labels.values())}
    return list(labels)

## this is if you dont have sub-folder for classes but classes are described in a labels.txt in the folder
def get_file_label_pairs_by_labelfile(folder, ignore_problems  = False):
    result = [y for x in os.walk(folder) for y in glob(os.path.join(x[0], '*.*'))]
    labels = open(folder + "/labels.txt", "r").read().split("\n")
    
    # if there is a labelnames.txt, replace the labels with the names defined in there
    if folder + "/labelnames.txt" in result:
        labels = replace_labels(labels, folder + "/labelnames.txt", ignore_problems = ignore_problems)
    print(f"There are effectively {len(np.unique(labels))} classes in the dataset.")
    
    # now creating the descriptor
    print("Creating now the descriptor")
    filenames = []
    for file in sorted(result):
        if file not in [folder + "/labels.txt", folder + "/labelnames.txt"]:
            filenames.append(file)
    if len(filenames) != len(labels):
        raise Exception(f"Folder contains {len(filenames)} files other than labels.txt but labels.txt defines labels for {len(labels)} files.")
    return [(filenames[i], labels[i]) for i in range(len(filenames))]


## this is if you have one sub-folder for each class
def get_file_label_pairs_by_subfolders(folder, ignore_problems = False):
    result = [y for x in os.walk(folder) for y in glob(os.path.join(x[0], '*.*'))]
    filenames = []
    labels = []
    for file in result:
        if not "labelnames.txt" in file:
            sub_folder = file[:file.rindex("/")]
            label = sub_folder[sub_folder.rindex("/") + 1:]
            filenames.append(file)
            labels.append(label)
        
    # if there is a labelnames.txt, replace the labels with the names defined in there
    if folder + "/labelnames.txt" in result:
        labels = replace_labels(labels, folder + "/labelnames.txt", ignore_problems = ignore_problems)
    print(f"There are effectively {len(np.unique(labels))} classes in the dataset.")
    if len(filenames) != len(labels):
        raise Exception(f"Folder contains {len(filenames)} files other than labels.txt but labels.txt defines labels for {len(labels)} files.")
    return [(filenames[i], labels[i]) for i in range(len(filenames))]



## this will create the desired folder from a description
def prepare_dataset(file_label_pairs, folder_target):
    
    Path(folder_target).mkdir(parents=True, exist_ok=True)
    
    # create images
    dataset_as_list = []
    img_dir = folder_target + "/images"
    for from_file, label in tqdm(file_label_pairs):

        # create folder for target file
        Path(img_dir).mkdir(parents=True, exist_ok=True)

        # prepare and write image
        to_file = from_file[from_file.rindex("/"):]
        to_path = img_dir + to_file
        dataset_as_list.append([to_file[1:], label, ""])
        image = cv2.imread(from_file)
        squared_image = perfect_square(image)
        resized_image = resize_128(squared_image)
        cv2.imwrite(to_path, resized_image)
        print(f"Converted {from_file} to {to_path}")
    
    # analysis of the categories
    df = pd.DataFrame(dataset_as_list, columns=["FILE_NAME", "CATEGORY", "SUPER_CATEGORY"])
    label_distributions = [len(group) for gIndex, group in df.groupby("CATEGORY")]
    
    # write label file
    labelfile = folder_target + "/labels.csv"
    df.to_csv(labelfile, index=False)
    print(f"Label file written to {labelfile}")
    
    # write info.json
    descriptor = {
        "dataset_name" : str(folder_target),
        "dataset_description" : "Description of " + str(folder_target),
        "total_categories" : len(label_distributions),
        "total_super_categorie" : 0,
        "uniform_number_of_images_per_category" : min(label_distributions) == max(label_distributions), 
        "images_per_category" : min(label_distributions),
        "has_super_categories" : False,
        "image_column_name" : "FILE_NAME",
        "category_column_name" : "CATEGORY",
        "super_category_column_name" : "SUPER_CATEGORY",
        "images_in_sub_folder" : True,
        "csv_with_tab": False
    }
    infofile = folder_target + "/info.json"
    json.dump(descriptor, open(infofile, "w"), indent=4)
    print(f"info.json written to {infofile}")


def prepare_dataset_from_labelfile(folder_src, folder_target = None, ignore_problems = False):
    if folder_target is None:
        folder_target = folder_src + "_resized"
    prepare_dataset(get_file_label_pairs_by_labelfile(folder_src, ignore_problems = ignore_problems), folder_target)
    
def prepare_dataset_from_subfolders(folder_src, folder_target = None, ignore_problems = False):
    if folder_target is None:
        folder_target = folder_src + "_resized"
    prepare_dataset(get_file_label_pairs_by_subfolders(folder_src, ignore_problems=ignore_problems), folder_target)
    
def plot_classes(folder, num_samples_per_class = 10, size_multiplier = 5):
    
    labels = pd.read_csv(folder + "/labels.csv")
    labelset = np.unique(labels["CATEGORY"])

    for label, dfLabel in labels.groupby("CATEGORY"):
        

        figwidth = num_samples_per_class
        figheight = 1
        figsize = (figwidth * size_multiplier, (figheight + 2.5) * size_multiplier)
        _, axarr = plt.subplots(figheight, figwidth, figsize=figsize)
        # Images are between -1 and 1.
        
        print(label)
        for i, filename in enumerate(dfLabel.sample(num_samples_per_class)["FILE_NAME"].values):
            image = mpimg.imread(folder + "/images/" + filename)
            axarr[i].imshow(image)
            axarr[i].set_title(filename)
            axarr[i].set(xticks=[], yticks=[])
        plt.show()