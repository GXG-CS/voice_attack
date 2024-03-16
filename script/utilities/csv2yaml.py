import csv
import yaml

# Define the CSV input file path and the YAML output file path
input_csv_file_path = 'C:\Users\GXG\Research\voice_attack\raw_data\WiSec\unmonitor_100\query_list_and_echo_times_100.csv'
output_yaml_file_path = 'C:\Users\GXG\Research\voice_attack\config\WiSec_unmonitor.yaml'

# Prepare to read the CSV file
with open(input_csv_file_path, mode='r', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    
    # Create a list to hold the YAML data structure
    yaml_list = []
    
    # Process each row in the CSV file
    for row in csv_reader:
        # Append a dictionary for each command to the list
        yaml_list.append({
            'subfolderName': f"simple_{csv_reader.line_num - 1}",  # Adjust based on your naming scheme
            'voiceCommand': row['Query'],
            'EchoIP': '10.0.0.159'
        })
        
    # Remove the last item if your setup requires it
    # yaml_list.pop()

# Write the list to a YAML file
with open(output_yaml_file_path, 'w', encoding='utf-8') as yaml_file:
    yaml.dump(yaml_list, yaml_file, default_flow_style=False, sort_keys=False)

print(f"YAML file has been created at: {output_yaml_file_path}")
