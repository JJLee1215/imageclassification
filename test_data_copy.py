'''
    Code to select specific devices and dates from the original images and copy them to the test images folder
'''

from datetime import datetime, timedelta
import os
import shutil
import sys

def generate_date_list(start_date, end_date):
    # Convert strings to datetime objects
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Create a list of dates
    date_list = []
    current_date = start
    while current_date <= end:
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)
    
    return date_list

# Example usage
mvpc_id = 'mvpc_1003'
data_start = '2024-10-05'
data_end = '2024-10-07'

date_list = generate_date_list(data_start, data_end)
print('date_list : ', date_list)
# Define source and target paths relative to the current working directory
base_source_path = os.path.join("..", "MVPC_원본", mvpc_id)
print('base_source_path : ', base_source_path)

target_path = os.path.join("data", "Test", "Test_image")

# Ensure target directory exists
os.makedirs(target_path, exist_ok=True)

# Iterate through the dates and copy .png files
for date in date_list:
    source_path = os.path.join(base_source_path, date)
    print(source_path)
    if os.path.exists(source_path):  # Check if the source folder exists
        for file_name in os.listdir(source_path):
            if file_name.endswith('.jpg'):  # Check if the file is a .png file
                source_file = os.path.join(source_path, file_name)
                target_file = os.path.join(target_path, file_name)
                
                # Copy the file
                shutil.copy2(source_file, target_file)
                print(f"Copied: {source_file} to {target_file}")
    else:
        print(f"Source folder does not exist: {source_path}")
