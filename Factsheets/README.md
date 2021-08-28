# Factsheets
Repository for generating Factsheets from dataset exploration for MetaDL Competition 2021


<br>

# ❗ Important
Make sure your dataset is in the required format before generating Factsheets.   
Check [Data Format for Factsheet Generation](../DataFormat/)

Some of the arguments related to dataset and not related to the experiment are now in `info.json` file of the dataset.

<br>
<br>

## Step 1 - Generate Factsheet report results

To generate Factsheet results, you can use either `generate_factshee_results.ipynb` or `generate_factsheet_results.py` 


### Ipython Notebook 
(skip this if you want to use python script)

In the notebook file `generate_factsheet_results.ipynb`, you have to set some variable in the cell under the heading **Settings** and then run the whole notebook to the end.

- `DATASET_PATH` : Path of the dataset which contains images, labels.csv and info.json
- `PREDICTIONS_PATH`: Path of the directory where the results of this experiments will be saved logs.txt, super_categories.txt and one directory for each super_category which contains categories.txt, categories_auc.txt, logs.txt, some figures etc 
- `CATEGORIES_TO_COMBINE` : number of categories to combine to make a super-category or a classification task (Default: *5*)
- `IMAGES_PER_CATEGORY` : number of images per category (Default: *20*)
- `MAX_EPISODES` : maximum limit on episodes/super-categories (Default: *None*)
- `USE_NORMALIZATION` : to normalize images in the way neural network is pretrained on ImageNet  (Default: *False*)
- `GENERATE_IMAGESHEET` : to generate an imagesheet : a pdf document with all the images per category (Default: *False*)

#### ⚠️ DEBUG 
- `DEBUG_MODE` : flag to activate debug mode (Default: *False*)
- `DEBUG_SUPER_CATEGORIES` : comma separated string with debug super-category names (Default: *None*)

#### ❗ DO NOT CHANGE
- `TRUE_SUPER_CATEGORIES` : variable configured in the notebook to generate True or Random Super-Categories *(Do not change)*
- `SEED` : seed for generating super-categories by the same random combination of categories *(Do not change)*

<br>


#### Examples of DEBUG_SUPER_CATEGORIES
(The debug super categories should be based on already generated Preliminary reports. It should be comma separated names of true super-categories or comma separated number of super-category if the Preliminary report is generated for random super-category)

***One debug category***  
- True Super Categories : 'bee'
- Random Super Categories: '4'

***Multiple debug category***  
- True Super Categories : 'bee, wasp, butterfly'
- Random Super Categories: '4, 9, 15'


<br>
<br>

### Python Script
(skip this if you want to use ipython notebook)

You can use the script `generate_factsheet_results.py` to generate the Preliminary report results by executing the following shell command with the required argumenets

```
python generate_factsheet_results.py \
--DATASET_PATH './data_set' \
--PREDICTIONS_PATH './experiment_1_results' \
```

***Optional Arguments***  
`--CATEGORIES_TO_COMBINE` (default: *5*)     
`--IMAGES_PER_CATEGORY` (default: *20*)  
`--MAX_EPISODES` (default: *None*)  
`--USE_NORMALIZATION` (default: *False*)  
`--GENERATE_IMAGESHEET` (default: False)    

***⚠️ Debug Arguments***  
`--DEBUG_MODE` (default: *False*)  
`--DEBUG_SUPER_CATEGORIES` (default: None) 



***Sample command with all Arguments*** 
```
python generate_factsheet_results.py \
--DATASET_PATH './data_set' \
--PREDICTIONS_PATH './experiment_1_results' \
--CATEGORIES_TO_COMBINE 6 \
--IMAGES_PER_CATEGORY 40 \
--MAX_EPISODES 50 \
--USE_NORMALIZATION \
--GENERATE_IMAGESHEET
```

***Sample command with Debug Arguments (True Super-Categories)*** 
```
python generate_factsheet_results.py \
--DATASET_PATH './data_set' \
--PREDICTIONS_PATH './experiment_1_results' \
--DEBUG_MODE \
--DEBUG_SUPER_CATEGORIES 'bee, wasp, butterfly'
```

***Sample command with Debug Arguments (Random Super-Categories)*** 
```
python generate_factsheet_results.py \
--DATASET_PATH './data_set' \
--PREDICTIONS_PATH './experiment_1_results' \
--DEBUG_MODE \
--DEBUG_SUPER_CATEGORIES '4, 9, 15'
```

<br>

#### Results
The results generated in this step contains the following files in the ***PREDICTIONS*** directory:

- logs.txt
- super_categories.txt
- one folder for each super_category (If you have 10 randomly generated super_categories, then you will see 10 folders named from 0-9)
- One super_category folder contains the following files:
    - logs.txt
    - categories.txt
    - categores_auc.txt
    - train.csv
    - valid.csv
    - train_results.png
    - confusion_matrix.png
    - auc.png
    - auc_histogram.png
    - roc_curves.png
    - sample_images.png
    - wrongly_classified_images.png
- descending_auc.png
- overall_auc_histogram.png
- imagesheet.pdf




<br>
<br>


## Step 2

Make sure to install **jinja2** and **pdfkit** before executing this step

Once you have your results from **Step 1**, you can now execute the python script `generate_pdf_report.py`.

Use the following command to create a PDF report from the generated results in the step above.

```
python generate_pdf_report.py \
--results_dir "./experiment_1_results" \
--title "Preliminary report Experiment # 1"
```

***Optional Arguments***  
`--keep_html` (default: False) : to get report both in `html` and `pdf` format


Use the following command to keep `html` report

```
python generate_pdf_report.py \
--results_dir "./experiment_1_results" \
--title "Preliminary report Experiment # 1" \
--keep_html
```

This script will generate a pdf report using the html template `template.html`. The pdf file will be stored in a newly created directory with the name ***report_files***.

The PDF will have a summary of the results in a table and then individual results of super-categories/classification tasks.



<br>
<br>

## ⚠️ Note
❗ The categories/classes are combined in a way in Step 1 that no category is repeated in super-categories.  
❗ The images in Preliminary report may not be the same as sample images in the experiment if you try it with a csv with exact number of images per category and more than required images per category. For example a csv with 40 images per category (with required images = 40) will have the sample images in experiment in imagesheet but a csv with 100 images per category (with required images = 40) may show different images in sample images and imagesheet.



## Troubleshooting

* Be aware that the proper installation of `pdfkit` can need installing `wkhtmltopdf`. Check this https://github.com/JazzCore/python-pdfkit/wiki/Installing-wkhtmltopdf 

For example, if you are on Debian / Ubuntu:

```bash
apt-get update
apt-get install wkhtmltopdf
```

* If the pdf report files cannot be automatically generated, you can keep the html report files (use the option `--keep_html` for `generate_pdf_report.py`) and convert them to pdf manually, for example via the print functionality of Chrome.
* You might encounter an issue of `Image size of ...x... pixels is too large. It must be less than 2^16 in each direction` if you have large number of classes. This problems comes from the generation of `descending_auc.png` in the function `generate_overall_auc_histogram_and_desc_auc_plot` of `generate_factsheet_results.py`. You can decrease the dpi in order to overcome this issue (decrease `X` in the line `fig.savefig(descending_categoris_auc_path, dpi=X)`).
