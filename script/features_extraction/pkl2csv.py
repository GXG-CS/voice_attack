import os
import pandas as pd

# Define the directory containing your .pkl files and the target directory for the .csv files
# pkl_directory = 'traffic_W_outgoing_features'  
pkl_directory = 'traffic_W_incoming_features'
# csv_directory = 'traffic_W_outgoing_features_csv'
csv_directory = 'traffic_W_incoming_features_csv'  

# Create the target directory if it doesn't exist
if not os.path.exists(csv_directory):
    os.makedirs(csv_directory)

# Process each .pkl file in the directory
for filename in os.listdir(pkl_directory):
    if filename.endswith('.pkl'):
        # Construct the full path to the .pkl file
        pkl_file_path = os.path.join(pkl_directory, filename)
        
        # Load the .pkl file
        with open(pkl_file_path, 'rb') as file:
            data = pd.read_pickle(file)
        
        # Convert the data to a DataFrame
        df = pd.DataFrame(data)
        
        # Construct the CSV file name (same as the .pkl file but with a .csv extension)
        csv_filename = os.path.splitext(filename)[0] + '.csv'
        csv_file_path = os.path.join(csv_directory, csv_filename)
        
        # Save the DataFrame to a CSV file
        df.to_csv(csv_file_path, index=False)
        
        print(f'{filename} has been converted to {csv_filename} and saved to {csv_directory}')

print('All .pkl files have been converted to .csv files.')
