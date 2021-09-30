# Add Benchmark Dataset to Meta-Album

To contribute a dataset to the Meta-Album benchmark, check the guidelines for dataset preparation and the reviewing criteria.  
*We encourage donors to first check examples of previously donated datasets and to* ***DIVERSIFY THE APPLICATION DOMAINS***

<br><br>

## Guidelines to prepare a benchmark dataset

1.  Format the data in the [File Format](../DataFormat/) and run the `check_data_format.py` script to check the data format.
    
2.  Run [Factsheets](../Factsheets/) experiment and prepare **Factsheet reports**.
    
3.  Convert the formatted data into TFRecords format by using the [Data Converter](../AutodlConverterModified/).
    
4.  Submit the TFRecords formatted data to the [MetaDL self-service](https://competitions.codalab.org/competitions/31280); record your Codalab ID and submission number.

5.  Upload to a location where it can be downloaded from the Internet (and record the DATA URL):
* the dataset in the File Format 
* the Factsheet reports
* the dataset in TFRecords format

6.  Fill the [Datasheet](https://forms.gle/pxm8qkpKywKjVobk8) for dataset, download it in PDF, and email it to `meta-album@chalearn.org`


<br><br>

## Reviewing criteria
The new submitted datasets will be reviewed on the following points:

1.  Relevance/usefulness to meta-learning
 
2.  Quality of answers provided in the Datasheet:
    -   Motivation
    -   Composition
    -   Collection Process
    -   Preprocessing/cleaning/labeling
    -   Uses
    -   Distribution
    -   Maintenance

3.  Compliance with the data formatting instructions and successful submission on [MetaDL self-service](https://competitions.codalab.org/competitions/31280)

4.  Performance on baselines algorithms


<br><br>


### Reviewers
The authors of Meta-Album have volunteered to review new additions to benchmark.  The review process should take 1 to 2 months.


<br><br>

### Contact: 
meta-album@chalearn.org
