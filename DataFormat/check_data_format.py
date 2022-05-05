
#--------------------------------
# Imports 
#--------------------------------
import os
import argparse
import json
import pandas as pd


#--------------------------------
# Arguments
#--------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, required=True,  
    help="""The directory path of the dataset""")

args = parser.parse_args()




#--------------------------------
# Settings
#--------------------------------
dataset_path = args.dataset_path
json_path = os.path.join(dataset_path,"info.json")
csv_path = os.path.join(dataset_path,"labels.csv")









#--------------------------------
# Errors
#--------------------------------
class DataFormatErrors:
    def FileError(self,file):
        raise IOError('[-] File Not Found : '+file)
        exit()
    def DirectoryError(self,directory):
        raise IOError('[-] Directory Not Found : '+directory)
        exit()
    def ColumnError(self,column):
        raise ValueError('[-] Column Not Found : '+column)
        exit()
err = DataFormatErrors()


#--------------------------------
# Check Directory and CSV
#--------------------------------

# Check Images Directory
if not os.path.exists(dataset_path):
    err.DirectoryError(dataset_path)

#Check JSON file
if not os.path.isfile(json_path):
    err.FileError(json_path)

#Check CSV file
if not os.path.isfile(csv_path):
    err.FileError(csv_path)




#--------------------------------
# Read JSON
#--------------------------------
f = open (json_path, "r")
info = json.loads(f.read())


#--------------------------------
# Read CSV
#--------------------------------
data = pd.read_csv(csv_path)





#--------------------------------
# Check Columns in CSV
#--------------------------------
csv_columns = data.columns

#Image 
if not info["image_column_name"] in csv_columns:
    err.ColumnError(info["image_column_name"])


#Category 
if not info["category_column_name"] in csv_columns:
    err.ColumnError(info["category_column_name"])



#Super Category 
if info["has_super_categories"]:
    if not info["super_category_column_name"] in csv_columns:
        err.ColumnError(info["super_category_column_name"])
    


#Image Folder
image_path = os.path.join(dataset_path,"images")
if not os.path.exists(image_path):
    err.DirectoryError(image_path)

print("###-------------------------------------###")
print("[+] Your dataset is in perfect format!")
print("[+] Now you can run factsheet to get some awesome results!")
print("###-------------------------------------###")