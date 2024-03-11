import os

# Specify the directory containing the small .txt files
source_directory = '../../raw_data/text_A/simple'

# Specify the path for the output file
output_file_path = 'combined_file.txt'

# Define a function to extract the numerical part from the filename
def extract_number(filename):
    # Split the filename by '_' and extract the last part as the number, then remove the '.txt' part
    parts = filename.split('_')
    number_part = parts[-1].split('.')[0]
    return int(number_part)

# Collect all .txt files in the specified directory
txt_files = [file for file in os.listdir(source_directory) if file.endswith('.txt')]

# Sort files by the numerical part in their filenames
txt_files.sort(key=extract_number)

# Open the output file in write mode
with open(output_file_path, 'w') as output_file:
    # Iterate over each file
    for txt_file in txt_files:
        # Construct the full path to the file
        file_path = os.path.join(source_directory, txt_file)
        
        # Open and read the contents of the file
        with open(file_path, 'r') as file:
            content = file.readline().strip()  # Read the first (and only) line, removing any trailing newline
            
            # Write the content to the output file, followed by a newline
            output_file.write(content + '\n')

print(f"All contents have been successfully combined into {output_file_path}.")
