'''
This script is used to find zipped files, which have .gzip formatted, 
in a folder and unzipped to a destined directory for further data analysing.
'''
# Libraries used:
import os
import re
import gzip, shutil
import concurrent.futures
import time

t1 = time.perf_counter()
DESTINED_PATH_MAIN = 'D:/Hung/Data Science/Data Science Scheme/Customer Churn & Segmentation/Raw data/unzipped_datas/SAOKE/'
DESTINED_PATH_INFO = 'D:/Hung/Data Science/Data Science Scheme/Customer Churn & Segmentation/Raw data/unzipped_datas/INFO/'
GET_PATH = 'D:/Hung/Data Science/Data Science Scheme/Customer Churn & Segmentation/Raw data'

# Get customers statement:
dirpaths_main = []

# Get customers information:
dirpaths_info = []

# Define pattern to recognize string liked that has 8 digits:
pattern = '\d{8}'

# Finding customer statement files in a destined folder:
for dirpath, dirname, filenames in os.walk(GET_PATH):
    # Find out if the direction path has a pattern like '\d{8}'
    if re.findall(pattern, dirpath) != []:
        
        for filename in filenames:
            # Split elements of a file name:
            head = os.path.splitext(filename)[0]
            tail = os.path.splitext(filename)[1]
            # Customer statement file has string liked 'SAOKE'
            if tail == '.zip' and re.findall('[A-Z0-9_]*SAOKE', head):
                path_in = dirpath + '/' + filename
                path_out = DESTINED_PATH_MAIN + head + '_' + \
                           re.findall(pattern, dirpath)[0] + '.csv'
                dirpaths_main.append((path_in, path_out))

# Finding customer information files in a destined folder:
for dirpath, dirname, filenames in os.walk(GET_PATH):
    if re.findall(pattern, dirpath) != []:
        
        for filename in filenames:
            head = os.path.splitext(filename)[0]
            tail = os.path.splitext(filename)[1]

            # Customer information file has string liked 'TTKH'
            if tail == '.zip' and re.findall('[A-Z0-9_]*TTKH', head):

                path_in = dirpath + '/' + filename
                path_out = DESTINED_PATH_INFO + head + '_' + \
                           re.findall(pattern, dirpath)[0] + '.csv'
                dirpaths_info.append((path_in, path_out))

def unzipped_file(path):
    ''' To unzip a gzip-formated file to a destined directory.'''
    with gzip.open(path[0], 'r') as f_in,\
        open(path[1], 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

# Run concurrent with defined user function unzipped_file:
with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(unzipped_file, dirpaths_info)

with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(unzipped_file, dirpaths_main)

t2 = time.perf_counter()

print(f'It takes {t2 - t1} seconds for completing the task.')
print('Unzipping completed.')
