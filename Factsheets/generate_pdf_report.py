import os
import argparse
import base64
import jinja2
import pdfkit
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument("--results_dir", type=str, required=True,  help="""The directory which contains all the experiments results, 
    e.g the results of each super_category in separate directory, super_categories.txt and logs.txt""")

parser.add_argument("--title", type=str, required=True, help="""Title of the html and pdf file.""")

parser.add_argument("--output_dir", type=str, default="./", help="""The directory where the html and pdf file will be stored. Default is current directory""")

parser.add_argument("--keep_html", action="store_true", default=False, 
    help="""Whether keep the html reports from which pdf reports can be built. 
    This is useful if the conversion between html and pdf fails.""")


args = parser.parse_args()



#----------------------------
# Create Output Directory
#----------------------------
output_dir = os.path.join(args.output_dir, 'report_files')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

title = args.title

factsheet_output_html = os.path.join(output_dir, 'factsheet_report_{}.html'.format(title.replace(" ", "_")))
factsheet_output_pdf = os.path.join(output_dir, 'factsheet_report_{}.pdf'.format(title.replace(" ", "_")))


images_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), args.results_dir))

    

print("###########################")
print("Preparing HTML File")
print("###########################")
print()


#----------------------------
# Read Super Categories
#----------------------------
super_categories_file = os.path.join(args.results_dir, "super_categories.txt")
with open(super_categories_file, "r", encoding="utf-8") as f:
    super_categories = f.read().splitlines()



#----------------------------
# Read Logs
#----------------------------
overall_logs_file = os.path.join(args.results_dir, "logs.txt")
with open(overall_logs_file, "r") as f:
    logs = f.read().splitlines()

    

#----------------------------
# Statistics Logs
#----------------------------
total_super_categories = logs[0].split(" : ")[1]
total_categories = logs[1].split(" : ")[1]
categories_combined = logs[3].split(" : ")[1]




#----------------------------
# Overall Histogram
#----------------------------
over_all_auc_histogram_path = os.path.join(images_path, "overall_auc_histogram.png")
with open(over_all_auc_histogram_path, "rb") as image_file:
    over_all_auc_histogram = base64.b64encode(image_file.read()).decode('ascii')





#----------------------------
# Prepare Episodes
#----------------------------
episodes = []

for super_cat in super_categories:
    
    super_cat_dic = {}
    
    super_cat_dic['super_category'] = super_cat
    
    #---------------------------------
    # Single Super Category Logs
    #---------------------------------
    super_cat_logs_file = os.path.join(args.results_dir, super_cat, "logs.txt")
    with open(super_cat_logs_file, "r") as f:
        super_cat_logs = f.read().splitlines()

    super_cat_dic['total_images'] = super_cat_logs[0].split(" : ")[1]
    super_cat_dic['train_images'] = super_cat_logs[1].split(" : ")[1]
    super_cat_dic['valid_images'] = super_cat_logs[2].split(" : ")[1]
    super_cat_dic['training_time'] = super_cat_logs[3].split(" : ")[1]
    super_cat_dic['best_train_acc'] = super_cat_logs[4].split(" : ")[1]
    super_cat_dic['best_valid_acc'] = super_cat_logs[5].split(" : ")[1]
    super_cat_dic['AUC'] = super_cat_logs[6].split(" : ")[1]
    super_cat_dic['2*AUC-1'] = super_cat_logs[7].split(" : ")[1]
    super_cat_dic['standard_error'] = super_cat_logs[8].split(" : ")[1]
 
    
    
    #---------------------------------
    # Categoriees
    #---------------------------------
    super_cat_categories_file = os.path.join(args.results_dir, super_cat, "categories_auc.txt")
    with open(super_cat_categories_file, "r", encoding="utf-8") as f:
        categories_lines = f.read().splitlines()
     
    categories,categories_auc = [],[]
    for line in categories_lines:
        cat, auc = line.split(" : ")
        categories.append(cat)
        categories_auc.append(float(auc))

    
    super_cat_dic['categories'] = categories
    super_cat_dic['min_AUC'] = '{:.2f}'.format(round(np.min(categories_auc),2))
    super_cat_dic['max_AUC'] = '{:.2f}'.format(round(np.max(categories_auc),2))
    super_cat_dic['sd_AUC'] = '{:.2f}'.format(round(np.std(categories_auc),2))
    
    
    
    
    
    
    #---------------------------------
    # Accuracy and Loss Plot
    #---------------------------------
    accuracy_loss_plot_path = os.path.join(images_path, super_cat, "train_results.png")
    with open(accuracy_loss_plot_path, "rb") as image_file:
        super_cat_dic['accuracy_loss_plot'] = base64.b64encode(image_file.read()).decode('ascii')
    
    #---------------------------------
    # Confusion Matrix
    #---------------------------------
    confusion_matrix_plot_path = os.path.join(images_path, super_cat, "confusion_matrix.png")
    with open(confusion_matrix_plot_path, "rb") as image_file:
        super_cat_dic['confusion_matrix_plot'] = base64.b64encode(image_file.read()).decode('ascii')
    
    #---------------------------------
    # AUC Plot
    #--------------------------------
    auc_plot_path = os.path.join(images_path, super_cat, "auc.png")
    with open(auc_plot_path, "rb") as image_file:
        super_cat_dic['auc_plot'] = base64.b64encode(image_file.read()).decode('ascii')
    
    #---------------------------------
    # AUC Histogram
    #--------------------------------
    auc_histogram_plot_path = os.path.join(images_path, super_cat, "auc_histogram.png")
    with open(auc_histogram_plot_path, "rb") as image_file:
        super_cat_dic['auc_histogram_plot'] = base64.b64encode(image_file.read()).decode('ascii')
    
    #---------------------------------
    # ROC Curves
    #--------------------------------
    roc_curves_plot_path = os.path.join(images_path, super_cat, "roc_curves.png")
    with open(roc_curves_plot_path, "rb") as image_file:
        super_cat_dic['roc_curves_plot'] = base64.b64encode(image_file.read()).decode('ascii')
    
    
    #---------------------------------
    # Sample Images
    #--------------------------------
    sample_images_path = os.path.join(images_path, super_cat, "sample_images.png")
    with open(sample_images_path, "rb") as image_file:
        super_cat_dic['sample_images'] = base64.b64encode(image_file.read()).decode('ascii')
    
    #---------------------------------
    # Wrongly Classified Images
    #--------------------------------
    wrongly_classified_images_path = os.path.join(images_path, 
                                                              super_cat, "wrongly_classified_images.png")
    with open(wrongly_classified_images_path, "rb") as image_file:
        super_cat_dic['wrongly_classified_images'] = base64.b64encode(image_file.read()).decode('ascii')
    
    
    episodes.append(super_cat_dic)

# calculate the average AUC across episodes
average_AUC = 0
cnt = 0
for episode in episodes:
    average_AUC += float(episode["AUC"])
    cnt += 1
average_AUC /= cnt
average_AUC = "{:.2f}".format(average_AUC)
    
    
subs = jinja2.Environment(
    loader=jinja2.FileSystemLoader('./')
).get_template('template.html').render(title=title,
                                       total_super_categories=total_super_categories,
                                       total_categories=total_categories,
                                       categories_combined=categories_combined,
                                       over_all_auc_histogram=over_all_auc_histogram,
                                       episodes=episodes,
                                       average_AUC=average_AUC)




#FactSheet 
# lets write the substitution to a file
with open(factsheet_output_html,'w', encoding="utf-8") as f: 
    f.write(subs)
    
print("###########################")
print("HTML File is Ready")
print("###########################")
print()
  
    
print("###########################")
print("Converting to PDF")
print("###########################")
print()
    
#Convert HTML to PDF
options={
    'page-size': 'A4',
    'margin-top': '0.5in',
    'margin-right': '0.5in',
    'margin-left': '0.5in',
    'margin-bottom': '0.5in',
    'header-right': '[page]',
    'encoding':'utf-8'
        }
pdfkit.from_file(factsheet_output_html, factsheet_output_pdf, options = options)

print("###########################")
print("PDF File is Ready")
print("###########################")
print()

if not args.keep_html:
    os.remove(factsheet_output_html)






