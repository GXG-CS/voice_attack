import os

# Path to the directory containing the .wav files
directory_path = '/Users/xiaoguang_guo@mines.edu/Documents/voice_attack/script/data_collection/audioPlay_A'

# Path to the output text file
output_file_path = '/Users/xiaoguang_guo@mines.edu/Documents/voice_attack/script/data_collection/output.txt'

# Generate the expected set of file names
expected_files = {f"{i}.wav" for i in range(1, 201)}

# Collect the set of actual file names in the directory
actual_files = set(os.listdir(directory_path))

# Find missing files by difference
missing_files = expected_files - actual_files

# Write the numbers of missing files to a text file
with open(output_file_path, 'w') as file:
    for file_number in sorted(int(file_name.split('.')[0]) for file_name in missing_files):
        file.write(f"{file_number}\n")
