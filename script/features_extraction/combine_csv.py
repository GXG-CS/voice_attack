import pandas as pd
import os

# Define the paths to your incoming and outgoing folders
incoming_path = 'traffic_W_incoming_features_csv'  
outgoing_path = 'traffic_W_outgoing_features_csv'  

# Define the command type mappings based on the file names
command_types = {'simple': 1, 'skill': 2, 'stream': 3}

# List all files in the incoming folder (assuming the same files exist in the outgoing folder)
incoming_files = os.listdir(incoming_path)

# Initialize an empty DataFrame to store combined data
combined_data = pd.DataFrame()

# Process each file
for file_name in incoming_files:
    # Determine the command type from the file name
    for command in command_types:
        if command in file_name:
            command_type = command_types[command]
            break
    else:
        raise ValueError(f"Unknown command type in file name: {file_name}")

    # Construct the full file paths
    incoming_file = os.path.join(incoming_path, file_name)
    outgoing_file = os.path.join(outgoing_path, file_name)
    
    # Read the incoming and outgoing data
    incoming_df = pd.read_csv(incoming_file)
    outgoing_df = pd.read_csv(outgoing_file)

    # Check that both files have the same number of rows
    if len(incoming_df) != len(outgoing_df):
        raise ValueError(f"File {file_name} has a mismatch in row count between incoming and outgoing data.")

    # Rename the columns to indicate incoming or outgoing
    incoming_df.columns = [f"{col}_incoming" for col in incoming_df.columns]
    outgoing_df.columns = [f"{col}_outgoing" for col in outgoing_df.columns]
    
    # Combine the data
    combined_df = pd.concat([incoming_df, outgoing_df], axis=1)

    # Add a new column for the command type
    combined_df['command_type'] = command_type
    
    # Append the combined data to the main DataFrame
    combined_data = pd.concat([combined_data, combined_df])

# Reset the index of the combined DataFrame
combined_data.reset_index(drop=True, inplace=True)

# Save the combined data to a CSV file
combined_data.to_csv('combined_features_with_types.csv', index=False)

print("All files have been combined into combined_features_with_types.csv")
