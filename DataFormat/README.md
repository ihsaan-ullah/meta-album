# Important Instructions for the processed version of dataset:
the dataset should be in the following format:

1. A parent directory with your dataset name which consists of the files `labels.csv` and `info.json`
2. preprocessed images (128x128) in either the same parent directory or in a folder named `images` inside the parent directory

<br>
<br>

### `labels.csv`
It should have at least these 3 columns:

- FILE_NAME
- CATEGORY
- SUPER_CATEGORY  

(see example datasets)

You can choose different column names, then specify them in info.json.

<br>
<br>

### `info.json`
It should have some information about the dataset  
- dataset_name (string)
- dataset_description (string)
- total_categories (integer)
- total_super_categorie (integer)
- uniform_number_of_images_per_category (boolean)
- minimum_images_per_category (integer)
- median_images_per_category (float)
- maximum_images_per_category (integer)
- has_super_categories (boolean)
- image_column_name (string)
- category_column_name (string)
- super_category_column_name (string) 

(see example datasets)

<br>

It is recommended to use the above mentioned column names but ofcourse you can use different names. Change the names of cloumns in `info.json` file to adjust it according to your dataset.  
The Factsheeet generator and other related scripts will be updated to give you the freedom of keeping your own column names.

<br>
<br>

### Examples of insects dataset is given below:
```
insects/
├── labels.csv
├── info.json
├── images/
├──── insect1.jpg
├──── insect2.jpg
├──── insect3.jpg
```


<br>
<br>

### Examples of labels.csv files for insects dataset:
***With predefined super-category***

```
FILE_NAME,      CATEGORY,   SUPER_CATEGORY
insect1.jpg,    bee,        insect                   
insect2.jpg,    wasp,       insect
insect3.jpg,    butterfly,  insect
```

***Without predefined super-category***
```
FILE_NAME,      CATEGORY,   SUPER_CATEGORY
insect1.jpg,    bee,        NAN                   
insect2.jpg,    wasp,       NAN
insect3.jpg,    butterfly,  NAN
```

<br>
<br>

### Examples of info.json file for insects dataset:

```json
{
    "dataset_name" : "mini_insect_1",
    "dataset_description" : "mini insect example dataset # 1",
    "total_categories" : 4,
    "total_super_categorie" : 1,
    "uniform_number_of_images_per_category" : true, 
    "minimum_images_per_category" : 3,
    "median_images_per_category" : 3.0,
    "maximum_images_per_category" : 3,
    "has_super_categories" : true,
    "image_column_name" : "FILE_NAME",
    "category_column_name" : "CATEGORY",
    "super_category_column_name" : "SUPER_CATEGORY",
}
```


<br>
<br>

# Sample Datasets
You will find three sample dataset in this repository

1. `mini_insect_1` : mini dataset with *Super Categories*
2. `mini_insect_2` : mini dataset without *Super Categories*


<br>
<br>


# Check Data Format
To check if your dataset is correctly formatted before running the **Factsheet Script**, use the python script `check_data_format.py`.

You can run the script in the following way 

```
python check_data_format.py --dataset_path './mini_insect_1'
```



