import pandas as pd
import os

# Define the paths to your incoming and outgoing folders
incoming_path = 'dilawer11/simple_100_alexa/traffic_W_incoming_features_csv'
outgoing_path = 'dilawer11/simple_100_alexa/traffic_W_outgoing_features_csv'

# List all files in the incoming folder (assuming the same files exist in the outgoing folder)
incoming_files = os.listdir(incoming_path)

# Initialize an empty DataFrame to store combined data
combined_data = pd.DataFrame()

# Process each file
for file_name in incoming_files:
    # Attempt to extract a numeric label from the filename
    try:
        # Assuming the label is at the end of the filename before .csv, adjust as needed
        label = int(file_name.rstrip('.csv').split('_')[-1])
    except ValueError as e:
        print(f"Skipping {file_name}: unable to extract a valid numeric label.")
        continue

    # Construct the full file paths
    incoming_file = os.path.join(incoming_path, file_name)
    outgoing_file = os.path.join(outgoing_path, file_name)

    # Read the incoming and outgoing data
    incoming_df = pd.read_csv(incoming_file)
    outgoing_df = pd.read_csv(outgoing_file)

    # Optionally, check that both files have the same number of rows or handle differently
    # This section can be adjusted based on how you want to handle mismatches

    # Rename the columns to indicate incoming or outgoing
    incoming_df.columns = [f"{col}_incoming" for col in incoming_df.columns]
    outgoing_df.columns = [f"{col}_outgoing" for col in outgoing_df.columns]

    # Combine the data
    combined_df = pd.concat([incoming_df, outgoing_df], axis=1)

    # Add a new column for the label
    combined_df['label'] = label

    # Append the combined data to the main DataFrame
    combined_data = pd.concat([combined_data, combined_df], ignore_index=True)

# Save the combined data to a CSV file
combined_data.to_csv('combined_incoming_outgoing.csv', index=False)

print("All files have been combined into combined_incoming_outgoing.csv")
