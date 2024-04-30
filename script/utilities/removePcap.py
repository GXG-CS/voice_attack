import os

# Path to the text file containing the files to remove
file_path = "/Users/xiaoguang_guo@mines.edu/Documents/voice_attack/config/files_to_remove.txt"

# Root directory containing the subfolders
root_dir = "/Users/xiaoguang_guo@mines.edu/Documents/voice_attack/data_processed/vc_200/alexa/total3"

# Function to parse the file
def parse_file_to_remove(file_path):
    folders_files = {}
    with open(file_path, 'r') as file:
        for line in file:
            if '=' in line:
                folder, files = line.strip().split('=')
                files_list = [f"{file_name}.pcap" for file_name in files.split('.')]
                folders_files[folder] = files_list
    return folders_files

# Read the folders and files to remove
folders_files = parse_file_to_remove(file_path)

# Iterate over the dictionary and remove the specified files in each subfolder
for folder, files in folders_files.items():
    folder_path = os.path.join(root_dir, folder)
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        try:
            os.remove(file_path)
            print(f"Removed: {file_path}")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")
