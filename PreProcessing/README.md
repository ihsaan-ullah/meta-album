# Preprocessing
This Repository is for the Data Preporcessing.


### 1. Crop and Resize
The `crop_resize.py` python script can be used to crop images into perfect squares and then resize into 128x128 size with anti-aliasing filter using open-cv. 

<br><br>

### 2. 40-images per class sub-dataset extraction
To extract a sub-dataset from a big dataset already formatted in the [Data Format](../DataFormat/), `data-preparation.ipynb` notebook can be used.  
This notebook will randomly sample 40 images per class and it will copy the required images to the destination directory and it will also create the meta-data file `info.json` which should be manually configured before extracting the dataset.


