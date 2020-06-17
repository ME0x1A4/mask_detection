# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 17:52:14 2020

@author: micha
"""

import csv,re
import scipy.io as sio

INFILE = ".\Medical_Masks_Dataset\monk_labels.csv"
OUTFILE_Labeled = ""
OUTFILE_Full = "onlyFaces.mat"
IMAGE_PATH_INPUT = ""
LOCATION_PRE = "Medical_Masks_Dataset/images/"

OUTPUT_BUFFER_FULL = {"imageFilename":[],"face":[]}
OUTPUT_BUFFER_Labeled = []

with open(INFILE) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            #print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            tmp_id = row[0]
            tmp_data = re.split(" ",row[1])
            tmp_full = "["
            for i in range(int(len(tmp_data)/5)):
                if i != 0: tmp_full+=";"
                tmp_full+= tmp_data[0+i*5]+","+tmp_data[1+i*5]+","+tmp_data[2+i*5]+","+tmp_data[3+i*5]         
            
            tmp_full+="]"    
            line_count += 1
            
#            OUTPUT_BUFFER_FULL.append({"imageFilename":tmp_id,"face":tmp_full})
            OUTPUT_BUFFER_FULL["imageFilename"].append(LOCATION_PRE+tmp_id)
            OUTPUT_BUFFER_FULL["face"].append(tmp_full)
            
            
sio.savemat(OUTFILE_Full, {'fullData':OUTPUT_BUFFER_FULL})