#Code to Detect CMEs(python)


#------------------------------------------------------################################-------------------------------------------------------------------------------#
#Read the BLK file of SWIS data. Visualize the data structure.

import cdflib
from spacepy import pycdf

BLK_file = r"F:\Aditya-L1\Data\AL1_ASW91_L2_BLK_20250704_UNP_9999_999999_V02.cdf"
TH1_file = r"F:\Aditya-L1\Data\AL1_ASW91_L2_TH1_20250704_UNP_9999_999999_V02.cdf"
TH2_file = r"F:\Aditya-L1\Data\AL1_ASW91_L2_TH2_20250704_UNP_9999_999999_V02.cdf"

cdf_file_blk = cdflib.CDF(BLK_file)
cdf_file_th1 = cdflib.CDF(TH1_file)
cdf_file_th2 = cdflib.CDF(TH2_file)


info_BLK = cdf_file_blk.cdf_info()
# print(cdf_file_blk.keys())      #it is possible if we use pycdf.

info_TH1 = cdf_file_th1.cdf_info()

info_TH2 = cdf_file_th2.cdf_info()


#z_Variables files of different data
zVar_BLK = cdf_file_blk.cdf_info().zVariables
zVar_th1 = cdf_file_th1.cdf_info().zVariables
zVar_th2 = cdf_file_th2.cdf_info().zVariables

for i,j,k in zip(zVar_BLK,zVar_th1,zVar_th2):
    print(i,' ',j," ",k)

#--------------BLK_file_variable---------------------#

import numpy
import pandas as pd

# Load your CDF file (replace 'your_file.cdf' with the actual filename)
cdf_file = cdflib.CDF(cdf_file_blk)

# List of compressed variables you want to extract
zVariables = ['epoch_for_cdf_mod', 'proton_density', 'numden_p_uncer', 'proton_bulk_speed',
              'bulk_p_uncer', 'proton_xvelocity', 'proton_yvelocity', 'proton_zvelocity',
              'proton_thermal', 'thermal_p_uncer', 'alpha_density', 'numden_a_uncer',
              'alpha_bulk_speed', 'bulk_a_uncer', 'alpha_thermal', 'thermal_a_uncer',
              'spacecraft_xpos', 'spacecraft_ypos', 'spacecraft_zpos']

# Create a dictionary to store variable data
data = {}

for var in zVariables:
    try:
        data[var] = cdf_file.varget(var)
    except KeyError:
        print(f"Variable {var} not found in the CDF file.")

# Convert dictionary to pandas DataFrame
df = pd.DataFrame(data)

print(df.head())








#------------------------------------------------------------------#############################---------------------------------------------------------------------#
#Save the BLK_file parameter to csvs



import os
import cdflib
import pandas as pd


file_path = r'F:\Aditya-L1\Aditya_L1_files'
# List all files in the folder
files_in_folder = os.listdir(file_path)


BLK_files = []
for file in files_in_folder:
    if "BLK" in file and  "V02" in file:
        BLK_files.append(file)


output_folder = "F:\Aditya-L1\Aditya_L1_csvs"

zVariables =  ['epoch_for_cdf_mod', 'proton_density', 'numden_p_uncer', 'proton_bulk_speed', 
                   'bulk_p_uncer', 'proton_xvelocity', 'proton_yvelocity', 'proton_zvelocity',
                   'proton_thermal', 'thermal_p_uncer', 'alpha_density', 'numden_a_uncer', 
                   'alpha_bulk_speed', 'bulk_a_uncer', 'alpha_thermal', 'thermal_a_uncer', 
                   'spacecraft_xpos', 'spacecraft_ypos', 'spacecraft_zpos']


for file_name in BLK_files:
    file_full_path = os.path.join(file_path, file_name)
    print(f"Processing: {file_full_path}")
    cdf_file = cdflib.CDF(file_full_path)
    data = {}

    # First, try to get all variables
    for var in zVariables:
        try:
            data[var] = cdf_file.varget(var)
        except KeyError:
            print(f"Variable {var} not found in {file_name}. Will fill with None later.")
            data[var] = None

    # Find the length of the main variable (usually 'epoch_for_cdf_mod')
    main_var = 'epoch_for_cdf_mod'
    length = len(data[main_var]) if data[main_var] is not None else 0

    # Now, fill missing variables with the correct length of None
    for var in zVariables:
        if data[var] is None:
            data[var] = [None] * length

    df = pd.DataFrame(data)

    # Save to CSV
    csv_file_name = file_name.replace('.cdf', '.csv')
    csv_path = os.path.join(output_folder, csv_file_name)
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")




#------------------------------------------------------------------##############################------------------------------------------------------------------#
#Now in the BLK csv there is a epoch parameter which indicates the time, so change it into UTC time data.





import os
import numpy as np
import pandas as pd
import cdflib

file_path = r'F:\Aditya-L1\Aditya_L1_files'
output_path = r'F:\Aditya-L1\Aditya_L1_csvs_modified\With_date'

if not os.path.exists(output_path):
    os.makedirs(output_path)

files_in_folder = os.listdir(file_path)
BLK_files = [file for file in files_in_folder if file.endswith('.cdf') and "BLK" in file and "V02" in file]

for file_name in BLK_files:
    file_full_path = os.path.join(file_path, file_name)


    print(f"Processing: {file_full_path}")
    cdf_file = cdflib.CDF(file_full_path)

    # Extract time variable and format as string with date and time
    time = cdflib.cdfepoch.to_datetime(cdf_file.varget("epoch_for_cdf_mod"))
    time_pd = pd.to_datetime(time)
    time_str = time_pd.strftime('%Y-%m-%d %H:%M:%S')

    def clean_data(varname):
        data = cdf_file.varget(varname)
        data = np.where(data == -1e31, np.nan, data)
        return data

    data_dict = {
        "Time_UTC": time_str,
        "Proton_Density_[#/cm3]": clean_data("proton_density"),
        "Proton_Bulk_Speed_[km/s]": clean_data("proton_bulk_speed"),
        "Proton_Thermal_Speed_[km/s]": clean_data("proton_thermal"),
        "Proton_XVelocity_[km/s]": clean_data("proton_xvelocity"),
        "Proton_YVelocity_[km/s]": clean_data("proton_yvelocity"),
        "Proton_ZVelocity_[km/s]": clean_data("proton_zv
elocity"),
        "Alpha_Density_[#/cm3]": clean_data("alpha_density"),
        "Alpha_Bulk_Speed_[km/s]": clean_data("alpha_bulk_speed"),
        "Alpha_Thermal_Speed_[km/s]": clean_data("alpha_thermal"),
        "SC_X_Pos_GSE_[km]": clean_data("spacecraft_xpos"),
        "SC_Y_Pos_GSE_[km]": clean_data("spacecraft_ypos"),
        "SC_Z_Pos_GSE_[km]": clean_data("spacecraft_zpos"),
    }

    df = pd.DataFrame(data_dict)
    csv_filename = os.path.splitext(file_name)[0] + ".csv"
    csv_full_path = os.path.join(output_path, csv_filename)
    df.to_csv(csv_full_path, index=False)
    print(f"CSV saved to: {csv_full_path}")


###############------------------------------------#####################
# 1 min interval file making


import os
import pandas as pd

input_folder = r'F:\Aditya-L1\Aditya_L1_csvs_modified\With_date'
output_folder = r'F:\Aditya-L1\Aditya_L1_csvs_modified\one_min_interval_data'

# List all CSV files in the input folder
csv_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.csv')]

for file_name in csv_files:
    file_path = os.path.join(input_folder, file_name)
    print(f"Processing: {file_path}")
    
    # Read the CSV
    df = pd.read_csv(file_path)
    
    # Parse 'Time_UTC' to datetime
    df['Time_UTC'] = pd.to_datetime(df['Time_UTC'], format='%Y-%m-%d %H:%M:%S')

    
    # Set 'Time_UTC' as index for resampling
    df.set_index('Time_UTC', inplace=True)
    
    # Resample to 1-minute intervals, taking the mean of numeric columns
    df_1min = df.resample('1T').mean().reset_index()
    
    # If you want to keep non-numeric columns (e.g., take the first value in each minute), add:
    # df_1min = df.resample('1T').agg({'col1': 'first', 'col2': 'mean', ...}).reset_index()
    
    # Format 'Time_UTC' back to your preferred string format
    df_1min['Time_UTC'] = df_1min['Time_UTC'].dt.strftime('%d-%m-%Y %H-%M-%S')
    
    # Save to output folder
    output_path = os.path.join(output_folder, file_name)
    df_1min.to_csv(output_path, index=False)
    print(f"Saved resampled CSV to: {output_path}")





#----------------------------------------------------------#####################################--------------------------------------------------------------------#
#Deviding LASCO Coronagraph Data based on the proton's Bulk speed.



import pandas as pd
file_name = r"F:\Aditya-L1\Cactus_Catalog\processing\cmeff.csv"
output_path = r"F:\Aditya-L1\Cactus_Catalog\cmeff_v700.csv"

# Read the CSV
df = pd.read_csv(file_name)

# Remove leading/trailing spaces in column names, if any
df.columns = df.columns.str.strip()

# Filter: halo is not 0 and not blank (NaN)
filtered = df[(df['halo'].notna()) & (df['halo'] != 0)]

# Further filter: v > 450
filtered = filtered[filtered['v'] > 700]

# Save to new CSV
filtered.to_csv(output_path, index=False)




#----------------------------------------------------------------------#######################-------------------------------------------------------------------#
#Labelling the data based on the LASCO Coronagraph using speed time relation and Train and test the Model for prediction.

'''Calculating the date that CME reaches from 
 Sun Lower corrona to L1 point
'''

import pandas as pd
from datetime import timedelta


cactus_final = r"F:\Aditya-L1\Cactus_Catalog\cmeff_v700.csv"
# Constants
DISTANCE_KM = 149597870  # Distance in km

# Load CSV
df = pd.read_csv(cactus_final, parse_dates=['t0'])

# Function to calculate arrival date
def calculate_arrival(row):
    velocity = row['v']  # velocity in km/s
    if velocity > 0:
        seconds_needed = DISTANCE_KM / velocity
        time_needed = timedelta(seconds=seconds_needed)
        arrival_date = row['t0'] + time_needed
        return arrival_date
    else:
        return pd.NaT  # Not a Time, if velocity is zero or negative

# Apply the function to each row
df['arrival_date'] = df.apply(calculate_arrival, axis=1)

# Save or display the result
# print(df[['t0', 'v', 'arrival_date']])
# df.to_csv('output_with_arrival_dates.csv', index=False)



#-------------------------------------------------------------------#
'''Shorting BLK files from folder 
based on the calculated arrival time form previous
 and marking all of these things as CME.
'''


import os

# Example: Load your DataFrame (replace with your actual DataFrame loading code)
# df = pd.read_csv('your_dataframe.csv')

# Step 1: Extract date and format as yyyymmdd
df['date_str'] = pd.to_datetime(df['arrival_date']).dt.strftime('%Y%m%d')

# Step 2: List all files in the folder
folder_path = r'F:\Aditya-L1\Aditya_L1_csvs'  # replace with your folder path
all_files = os.listdir(folder_path)

# Step 3: Match dates with file names
CME_list = []
for date in df['date_str'].unique():
    for file in all_files:
        if date in file:
            CME_list.append(file)

# matched_files now contains all matching file names
print(CME_list)

#---------------------------------------------------------------#
'''Remaining file dumped at another list'''
Non_CME_list = [f for f in all_files if f not in CME_list]

#----------------------------------------------------------------#
'''Creating Trainng dataset '''

'''So In the data files it contains lots of nan values and the remaining
    so we replace this nan values based on the mean, median, mode:
    so I am taking the MEAN replaced data as a training and validation 
    
    Now Active CME is file is listed in CME List and Non active CME files are 
    listed in Non_CME list.
    
    But we have seperately processed data for generating mean median mode ,
    so I have to match the data based on the CME_List and Non_CME_List
    '''

mean_data = r"F:\Aditya-L1\Train_data\mean_data"




X_CME = []
Y_CME = []
X_Non_CME = []
Y_Non_CME = []

for file_name in os.listdir(mean_data):
    if file_name.endswith('.csv'):
        file_path = os.path.join(mean_data, file_name)
        try:
            arr = pd.read_csv(file_path, usecols=['Proton_Bulk_Speed_[km/s]']).values  # shape (N, 1)
        except ValueError:
            print(f"Column 'proton_density' not found in {file_name}")
            continue

        if file_name in CME_list:
            X_CME.append(arr)
            Y_CME.append(1)
        elif file_name in Non_CME_list:
            X_Non_CME.append(arr)
            Y_Non_CME.append(0)

'''
Due to the abonormal sized of data we have to take only fixeld sized data.
In this case maximum no of rows here (1440,1)'''


from collections import Counter

# 1. Get all lengths in both lists
lengths_cme = [arr.shape[0] for arr in X_CME]
lengths_non_cme = [arr.shape[0] for arr in X_Non_CME]

# 2. Find intersection of lengths present in both
common_lengths = set(lengths_cme) & set(lengths_non_cme)

if not common_lengths:
    raise ValueError("No common array lengths found between CME and Non-CME lists!")

# 3. For each common length, count total occurrences in both lists and pick the most common
length_counts = {}
for length in common_lengths:
    length_counts[length] = lengths_cme.count(length) + lengths_non_cme.count(length)

# 4. Pick the length with the highest combined count
target_length = max(length_counts, key=length_counts.get)

# 5. Filter both lists to keep only arrays with this length
X_CME_filtered = [arr for arr in X_CME if arr.shape[0] == target_length]
Y_CME_filtered = [label for arr, label in zip(X_CME, Y_CME) if arr.shape[0] == target_length]

X_Non_CME_filtered = [arr for arr in X_Non_CME if arr.shape[0] == target_length]
Y_Non_CME_filtered = [label for arr, label in zip(X_Non_CME, Y_Non_CME) if arr.shape[0] == target_length]

print(f"Selected common length: {target_length}")
print(f"CME arrays kept: {len(X_CME_filtered)}")
print(f"Non-CME arrays kept: {len(X_Non_CME_filtered)}")




#------------------------- applying Randomforest_Classifier-------------------#
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Assume X_CME, X_Non_CME, Y_CME, Y_Non_CME are already prepared as lists of arrays

# 1. Combine data and labels
X_all = X_CME_filtered + X_Non_CME_filtered  # List of arrays
y_all = Y_CME_filtered + Y_Non_CME_filtered  # List of labels

# 2. Flatten each (N, 1) array to (N,) and stack into a 2D array (samples, features)
X_flat = [x.flatten() for x in X_all]
X = np.stack(X_flat)  # shape: (num_samples, sequence_length)
y = np.array(y_all)   # shape: (num_samples,)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

