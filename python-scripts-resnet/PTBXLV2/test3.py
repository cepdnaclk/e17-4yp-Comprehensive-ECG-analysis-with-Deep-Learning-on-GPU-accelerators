import os

import pandas as pd


current_directory = os.getcwd()                             # /e17-4yp-Comp.../python-scripts-resnet/PTB-XL
# print(current_directory)
parent_directory = os.path.dirname(current_directory)       # /e17-4yp-Comp.../python-scripts-resnet
# print(parent_directory)
super_parent_directory =os.path.dirname(parent_directory)   # # /e17-4yp-Comp...
# print(super_parent_directory)

features_csv_path = os.path.join(super_parent_directory,  'data', 'ptb-xl+', 'features', '12sl_features.csv') 
statements_csv_path = os.path.join(super_parent_directory,  'data', 'ptb-xl+', 'labels', 'ptbxl_statements.csv')   

df = pd.read_csv(features_csv_path) 
statements_df = pd.read_csv(statements_csv_path)

print('\nInitial\n')
print(df)

# ecg_id = statements_df['ecg_id'].values[0]
# print(type(ecg_id))
# print(ecg_id)

# scp_codes = statements_df['scp_codes'].values[0]
# print(type(scp_codes))
# print(scp_codes)

# if 'NORM' not in scp_codes:
#     print('fales')
# else:
#     print('true')

rows_to_remove = [] 
rows_to_remove_norm = []

# Iterate through the rows
for index, row in df.iterrows():
    file_index = int(df['ecg_id'].values[index])
    folder_name = str(file_index // 1000).zfill(2)+'000' 
    file_name = str(file_index).zfill(5)+'_hr.hea'
    # ecg_record_path = os.path.join(self.super_parent_directory,  'data', 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1', 'records500', folder_name, file_name)
    ecg_record_path = os.path.join(super_parent_directory,  'data', 'ptb-xl', 'records500', folder_name, file_name)
    #print(ecg_record_path)

    # Check if the ecg_record_path exists
    if not os.path.exists(ecg_record_path):
        rows_to_remove.append(index)

# Remove rows where ecg_record_path does not exist
df.drop(rows_to_remove, inplace=True)

print('\nAfter removing non existing records\n')
print(df)

# Reset the DataFrame index if needed
df.reset_index(drop=True, inplace=True)

print('\nAfter restting the index\n')
print(df)

# Iterate through the rows to remove non normal ecg
for index, row in statements_df.iterrows():
    ecg_id = statements_df['ecg_id'].values[index]
    scp_codes = statements_df['scp_codes'].values[index]

    if 'NORM' not in scp_codes:
        rows_to_remove_norm.append(ecg_id)

print('\nAfter creating the non norm list\n')        
print(rows_to_remove_norm[0:10])

for index, row in df.iterrows():
    if row['ecg_id'] in rows_to_remove_norm:
        df.drop(index, inplace=True)

print('\nAfter removing non NORM\n')        
print(df)

# Reset the DataFrame index if needed
df.reset_index(drop=True, inplace=True)

print('\nAfter resetting index\n')
print(df)
