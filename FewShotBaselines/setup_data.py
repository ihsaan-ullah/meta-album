import os
import lzma
import tarfile
import shutil
import zipfile

file_to_dataset = {
    "Insects_FortyImagesPerCategory_Forty_Images_Per_Category.zip": "insects",
    "Plankton_FortyImagesPerCategory_Forty_Images_Per_Category.zip": "plankton",
    "plantvillage_plantvillage-formatted-image (1).zip": "plants",
    "MedicinalLeaf_medleaf-formatted-image.zip": "medleaf",
    "Texture_1_FortyImagesPerCategory_Forty_Images_Per_Category.zip": "texture1",
    "Texture_2_FortyImagesPerCategory_Forty_Images_Per_Category.zip": "texture2",
    "rsi-cb-128-remotesensing_rsicb128-formatted-image.zip": "rsicb",
    "resisc45-remotesensing_resisc45-formatted-image (1).zip": "resisc",    
    "OmniPrint_overview_OmniPrint_MetaDL_Ihsan_format_meta-mix_first_set.zip": "omniprint1",
    "OmniPrint_overview_OmniPrint_MetaDL_Ihsan_format_meta5-bis_first_set.zip": "omniprint2",
}

all_data = "publicdata.zip"
root_dir = "./data/"

assert os.path.exists(all_data), "Could not find {} in the current directory".format(all_data)

if not os.path.isdir(root_dir):
    os.mkdir(root_dir)

# unzip the alldata.zip
with zipfile.ZipFile(all_data, 'r') as zip_ref:
        zip_ref.extractall("./")

for zfile, dirname in file_to_dataset.items():
    print("Processing {} files".format(dirname))
    unzip_location = os.path.join(root_dir, dirname)
    if not os.path.isdir(unzip_location):
        os.mkdir(unzip_location)
    else:
        print("\tDirectory {} already existed. Not touching this and moving to the next one".format(unzip_location))
        continue

    # Check file extension (if .zip -> unzip, if .xz -> convert to tar and untar)
    extension  =zfile.split(".")[1]
    if extension.lower() == "zip":
        # Read zip file and extract it 
        with zipfile.ZipFile(zfile, 'r') as zip_ref:
            zip_ref.extractall(unzip_location)
    elif extension.lower() == "xz":
        with lzma.open(zfile) as f:
            with tarfile.open(fileobj=f) as tar:
                tar.extractall(unzip_location)
    else:
        print("Unknown file extension .{} for {}".format(extension, dirname))


    os.remove(zfile)

    # Make sure there now is a folder called images in the unzip location
    image_dir = os.path.join(unzip_location, "images")
    if not os.path.isdir(image_dir):
        folder_in_zip_loc = os.path.join(unzip_location, os.listdir(unzip_location)[0])
        files_to_move = os.listdir(folder_in_zip_loc)
        for f in files_to_move:
            # unpack the folder
            shutil.move(os.path.join(folder_in_zip_loc, f), os.path.join(unzip_location, f))
        shutil.rmtree(folder_in_zip_loc)
    print("\tSuccess.")
print("\n[*] Everything went well. Data sets are ready!")

    
