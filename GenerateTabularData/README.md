# Generate Tabular Data
Repository for generating tabular data for MetaDL Competition 2021


<br>

# ❗ Important
Make sure that your dataset and retrain_dataset are in the required format before generating tabular data.   
Check [Data Format for Preliminary Report Generation](../DataFormat/)


<br>
<br>

## How It works

To generate tabular data, you can use either `generate_tabular_data.ipynb` or `generate_tabular_data.py` 


### Ipython Notebook 
(skip this if you want to use python script)

In the notebook file `generate_tabular_data.ipynb`, you have to set some variable in the cell under the heading **Settings** and then run the whole notebook to the end.

- `DATASET_PATH` : Path of the dataset which contains images, labels.csv and info.json
- `TABULAR_PATH`: Path of the directory where the tabular data will bee saved

- `USE_NORMALIZATION` : to normalize images in the way neural network is pretrained on ImageNet  (Default: *False*)
- `RETRAIN` : to retrain the network on your data before generating tabular data (Default: *False*)
- `RETRAIN_DATASET_PATH` : the path to directory where the training images are stored (128x128x3) (Default: *None*)



<br>
<br>

### Python Script
(skip this if you want to use ipython notebook)

You can use the script `generate_tabular_data.py` to generate tabular by executing the following shell command with the required argumenets

```
python generate_tabular_data.py \
--DATASET_PATH './data_set' \
--TABULAR_PATH './data_set_tabular'
```

***Optional Arguments***   
`--USE_NORMALIZATION` (default: *False*)  
`--RETRAIN` (default: False)   
`--RETRAIN_DATASET_PATH` (default: None)    



***Sample command with all Arguments*** 
```
python generate_tabular_data.py \
--DATASET_PATH './data_set' \
--TABULAR_PATH './data_set_tabular' \
--USE_NORMALIZATION \
--RETRAIN \
--RETRAIN_DATASET_PATH './data_set_retrain'
```


<br>

#### Results
The results generated in this step contains only one `labels.csv` files in the ***TABULAR*** directory which contains 512 features per image as columns and one more column `CATEGORY` for label of the input.





<br>
<br>



