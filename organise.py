import os
from config import results_directory
import shutil

# Define paths
folders_to_move = [
    'images',
    'model',
    os.path.join('data', 'processed')
]

files_to_move = [
    os.path.join('data', 'data_params.csv'),
    os.path.join('data', 'train_summary.csv')
]

# Create the results directory if it doesn't exist
os.makedirs(results_directory, exist_ok=True)

# Move folders
for folder in folders_to_move:
    if os.path.isdir(folder):
        dest_folder = os.path.join(results_directory, os.path.basename(folder))
        shutil.move(folder, dest_folder)
        print(f"Moved folder '{folder}' to '{dest_folder}'")
    else:
        print(f"Folder not found: {folder}")

# Move files
for file in files_to_move:
    if os.path.isfile(file):
        dest_file = os.path.join(results_directory, os.path.basename(file))
        shutil.move(file, dest_file)
        print(f"Moved file '{file}' to '{dest_file}'")
    else:
        print(f"File not found: {file}")