import pandas as pd

# Define file paths
incoming_path = 'google/no_trim/incoming.csv'
outgoing_path = 'google/no_trim/outgoing.csv'
# raw_path = 'raw.csv'

# Load the datasets
incoming_df = pd.read_csv(incoming_path)
outgoing_df = pd.read_csv(outgoing_path)
# raw_df = pd.read_csv(raw_path)

# Function to rename columns except for 'label'
def rename_columns(df, prefix):
    return df.rename(columns={col: f"{prefix}_{col}" if col != 'label' else col for col in df.columns})

# Rename columns
incoming_df = rename_columns(incoming_df, 'incoming')
outgoing_df = rename_columns(outgoing_df, 'outgoing')
# raw_df = rename_columns(raw_df, 'raw')

outgoing_df = outgoing_df.drop(columns=['label'])
# raw_df = raw_df.drop(columns=['label'])

# Concatenate incoming and outgoing, keeping only one 'label' column
incoming_outgoing_df = pd.concat([incoming_df, outgoing_df], axis=1)

# Save the concatenated DataFrame to a new CSV file
incoming_outgoing_df.to_csv('IO.csv', index=False)
print("Saved concatenated incoming and outgoing to IO.csv")

# Concatenate incoming, outgoing, and raw, keeping only one 'label' column
# all_df = pd.concat([incoming_df, outgoing_df, raw_df], axis=1)

# Save the full concatenated DataFrame to a new CSV file
# all_df.to_csv('IOR.csv', index=False)
# print("Saved concatenated incoming, outgoing, and raw to IOR.csv")
