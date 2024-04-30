import os

# Define the root directory path
root_dir = '/Users/xiaoguang_guo@mines.edu/Documents/voice_attack/data_processed/vc_200/alexa/total3'  # Replace with your actual root folder path

# Function to check if the file name (without extension) ends with a number from 201 to 249
def is_target_file(filename):
    if not filename.endswith('.pcap'):
        return False
    number_part = filename.split('.')[0]  # Assuming filename format is "{number}.pcap"
    try:
        number = int(number_part)
        return 201 <= number <= 249
    except ValueError:
        return False

# Walk through the directory
for dirpath, dirnames, files in os.walk(root_dir):
    for file in files:
        if is_target_file(file):
            # Construct the full file path
            file_path = os.path.join(dirpath, file)
            # Delete the file
            os.remove(file_path)
            print(f"Deleted {file_path}")
