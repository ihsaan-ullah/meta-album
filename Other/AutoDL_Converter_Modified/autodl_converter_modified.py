"""
Define the AutoDLConverter.

Example of run (on a personnal computer): 
python autodl_converter.py --dataset_name cifar100 \
     --dataset_root /Users/adrian/Downloads\
         /cifar100/ --records_path /Users/adrian/Documents/records

-------------------------------------------------------------------------------
Testing :
--------- 
    To test the data generation pipeline with the newly generated tfrecords,
    refer to the dataset_test.py script.
    
-------------------------------------------------------------------------------
TODO (ebadrian): 
    - Test other AutoDl datasets (than Cifar100)

"""
import os 
import csv
import collections
import itertools
import sys

from absl import app
from absl import flags
from absl import logging
from meta_dataset.dataset_conversion.dataset_to_records import DatasetConverter
from meta_dataset.dataset_conversion.dataset_to_records import write_tfrecord_from_image_files
from meta_dataset.dataset_conversion.dataset_to_records import gen_rand_split_inds
from meta_dataset.data import learning_spec

#-------------------------------
import json
import pandas as pd
#-------------------------------

#-------------------------------
data = None
info = None
#-------------------------------


FLAGS = flags.FLAGS


flags.DEFINE_string('dataset_root', './cifar100',
                       'Path to the directory containing the AutoDL dataset.')

flags.DEFINE_string('records_path', './',
                       'Path to store the newly generated records.')

class AutoDLConverter(DatasetConverter):
    """Convert the AutoDL datasets to the Meta-dataset format. 
    """
    def __init__(self, *args, **kwargs):
        """Initialize an AutoDLConverter."""
        # Make has_superclasses default to False for the AutoDL datasets.
        if 'has_superclasses' not in kwargs:
            kwargs['has_superclasses'] = False
        super(AutoDLConverter, self).__init__(*args, **kwargs)

        
        

    def create_dataset_specification_and_records(self):
        """ Write the tfrecords following the Meta-dataset logic. Each record
        file will contain examples of one class. 
        """

        
        splits = self.get_splits(force_create= True) # calls create_splits()
        
        # Get the names of the classes assigned to each split.
        train_classes = splits['train']
        valid_classes = splits['valid']
        test_classes = splits['test']


        self.classes_per_split[learning_spec.Split.TRAIN] = len(train_classes)
        self.classes_per_split[learning_spec.Split.VALID] = len(valid_classes)
        self.classes_per_split[learning_spec.Split.TEST] = len(test_classes)

        filepaths = collections.defaultdict(list)

        for i,row in data.iterrows():
            filepaths[row[info['category_column_name']]].append(os.path.join(self.data_root,row[info['image_column_name']]))
        
        keys = list(filepaths.keys())
        keys_len = len(keys)


        
        # Reading label nams from label.name
        # with open(labelname_path) as f:
        #     label_names = f.read().splitlines()

        
        
        names2trueidx = {
            keys[i] : i for i in range(keys_len) }

        
 
        logging.debug('names2trueidx : {}'.format(names2trueidx))

        all_classes = list(
            itertools.chain(train_classes, valid_classes, test_classes))
        
        
       
        # Class IDs are constructed in such a way that
        #   - training class IDs lie in [0, num_train_classes),
        #   - validation class IDs lie in
        #     [num_train_classes, num_train_classes + num_validation_classes), and
        #   - test class IDs lie in
        #     [num_train_classes + num_validation_classes, num_classes).

        for class_id, class_label in enumerate(all_classes):
            logging.info('Creating record for class ID %d (%s)...', class_id, class_label)

         
            
            original_id = names2trueidx[class_label]

            
            # class_paths = filepaths[original_id]
            class_paths = filepaths[class_label]

            

            class_records_path = os.path.join(
                self.records_path, 
                self.dataset_spec.file_pattern.format(class_id))

            
            
           
            self.class_names[class_id] = class_label
            self.images_per_class[class_id] = len(class_paths)

           


            
            # Create and write the tf.Record of the examples of this class.
            write_tfrecord_from_image_files(
                class_paths, class_id, class_records_path)


    def create_splits(self):
        """ Creates the splits for the AutoDL Dataset. It returns a dictionnary
        which values correspond to the class names associated to the key 
        (split).
        """

    

        
        filepaths = collections.defaultdict(list)
       
        for i,row in data.iterrows():
            filepaths[row[info['category_column_name']]].append(row[info['image_column_name']]) 
        
        keys = list(filepaths.keys())

        num_classes = len(keys)

        class_names = keys


        logging.debug('Verifying classes in create_dataset[...] function ...\n')
        logging.debug('Total number of classes detected in labels.csv : \
             {}'.format(num_classes))
        logging.debug('Detected classes names : {}'.format(class_names))


        # Split into train, validation and test splits that have 70% / 15% / 15%
        # of the data, respectively.
        num_trainval_classes = int(0.85 * num_classes)
        num_train_classes = int(0.7 * num_classes)
        num_valid_classes = num_trainval_classes - num_train_classes
        num_test_classes = num_classes - num_trainval_classes

        
        train_inds, valid_inds, test_inds = gen_rand_split_inds(
            num_train_classes, num_valid_classes, num_test_classes)

        splits = {
            'train' : [class_names[i] for i in train_inds],
            'valid' : [class_names[i] for i in valid_inds],
            'test' : [class_names[i] for i in test_inds]
        }

        

        return splits
        


def main(argv):
    """ Convert Cifar100 from the AutoDL format to the Meta-dataset format.
    convert_dataset() : Call create_dataset_specification_and_records() and 
                        write <dataset_spec>.json.
    """
    del argv
    # dataset_name = FLAGS.dataset_name
    autodl_dataset_root = FLAGS.dataset_root
    path_to_records = FLAGS.records_path


    #-------------------------------
    global data
    global info
    #-------------------------------



    #-------------------------------
    # Load Info JSON
    #-------------------------------
    JSON_PATH = os.path.join(autodl_dataset_root, "info.json")
    f = open (JSON_PATH, "r")
    info = json.loads(f.read())


    #-------------------------------
    # Setup variables
    #-------------------------------
    dataset_name = info['dataset_name']
    images_in_sub_folder = info['images_in_sub_folder']
    csv_with_tab = info['csv_with_tab']

    images_path = os.path.join(autodl_dataset_root, "images") if images_in_sub_folder else  autodl_dataset_root 
    
    #-------------------------------
    # Load CVS
    #-------------------------------
    CSV_PATH = os.path.join(autodl_dataset_root, "labels.csv")
    if csv_with_tab:
        data = pd.read_csv(CSV_PATH, sep="\t", encoding="utf-8") 
    else:   
        data = pd.read_csv(CSV_PATH)


    

    Dataset_converter = AutoDLConverter(
        name = dataset_name,
        data_root = images_path,
        has_superclasses = False,
        records_path=path_to_records
    )
    
    Dataset_converter.convert_dataset()

    logging.debug('\n Split created with the following attributes : \n')
    logging.debug('IMAGES per class : \
         {}'.format(Dataset_converter.images_per_class))
    
    logging.debug('Dataset Name  : {}'.format(Dataset_converter.name))
    logging.debug('Classes per split : \
         {}'.format(Dataset_converter.classes_per_split))
    logging.debug('Class names  : {}'.format(Dataset_converter.class_names))
    logging.debug('Records path  : {}'.format(Dataset_converter.records_path))
    logging.debug('File pattern  : {}'.format(Dataset_converter.file_pattern))

if __name__ == '__main__':
    app.run(main)
    