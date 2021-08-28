# MetaDL Modified Data Converter (For CSV File)
Modified AutoDL converter to generate tfrecords

<br>
<br>

# Important
This repository is directory related to the AutoDL data conversion `LLDL` repository. You can find the original `autodl_converter.py` there.

<br>
<br>

# Background
The `autodl_converter.py` expects the class of the image in the name of the file.
For example: an image name ***butterfly_00001.jpg*** consists of the image number i.e ***00001*** and the class ***butterfly***

<br>
<br>

# How the modified file works?
The file `autodl_converter_modified.py` expects the dataset in the ***DataFormat*** mentioned in this repository
Check this : [DataFormat](../DataFormat/)

<br>
<br>

# How to convert data?
You can run the file with the following command:

```
python autodl_converter_modified.py \
--dataset_root DIRECTORY_PATH_OF_DATASET \
--records_path DIRECTORY_PATH_TO_STORE_CONVERTED_DATA
```

<br>
<br>

# Example Dataset
Please check the example `mini_insect` dataset.

An example command to convert ***mini_insect*** dataset is given below:

```
python autodl_converter_modified.py \
--dataset_root ./mini_insect \
--records_path ./mini_insect_converted
```

<br>
<br>

### Credits
Adrian EL BAZ [https://github.com/ebadrian]  
