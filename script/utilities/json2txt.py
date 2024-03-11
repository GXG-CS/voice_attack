import json
import os

def generate_txt_files(json_path, output_directory):
    # Create the output directory if it doesn't already exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Load the JSON data
    with open(json_path, 'r') as file:
        data = json.load(file)

    # Generate a .txt file for each element in the JSON array
    for i, item in enumerate(data):
        # Format the filename with a leading number for ordering
        file_name = f"{output_directory}/simple_{i+1}.txt"
        # Write the modified invoke phrase to the file
        with open(file_name, 'w') as txt_file:
            txt_file.write(f"Echo, {item['invokePhrase']}")

# Specify the path to the JSON file and the output directory
json_path = '../../raw_data/dilawer11/simple_50_alexa/simple_50_alexa.json'  
output_directory = '../../raw_data/dilawer11/simple_50_alexa/text_A' 

# Call the function with your paths
generate_txt_files(json_path, output_directory)

print("TXT files have been generated successfully.")
