

import shutil
import os

import pandas as pd


def evaluation(forecast_file_csv, ann_files_save_folder):
    # my forecast file df has columns:  sentence	true_label	pred_label	drug	span_start	span_end	anno_type	filename
    df=pd.read_csv(forecast_file_csv)
    
    #since I am appending to the .ann files, i delete the folder at the beginning of each run
    try:shutil.rmtree(ann_files_save_folder)
    except:pass
    try:os.mkdir(ann_files_save_folder)
    except:pass

    #change forecasted csv file to .ann file
    for i in range(len(df)):
        filename = df['filename'][i]
        save_file = open(os.path.join(ann_files_save_folder, filename.split('/')[-1]), 'a')
        # line1: write text in the form: 'T1\tNoDisposition 821 827\tProzac\n'
        line1 = '{}\t{} {} {}\t{}\n'.format(df['anno_type'][i],
                                                df['pred_label'][i],
                                                df['span_start'][i],
                                                df['span_end'][i],
                                                df['drug'][i])
        #line2: write text in the form E2\tDisposition:T2 \n
        line2 = '{}\t{}:{} \n'.format(df['anno_type'][i].replace('T', 'E'),
                                          df['pred_label'][i], df['anno_type'][i])
        save_file.write(line1)
        save_file.write(line2)



df='prediction_results_200_release2_bert_large.csv'
annFol='../' + df.split('.')[-2] + '/'
evaluation(df,annFol)