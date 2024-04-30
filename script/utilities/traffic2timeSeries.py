import pandas as pd
import os
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor

def clean_timestamps(input_file, output_file, direction):
    # Load the CSV file
    df = pd.read_csv(input_file)

    # Remove the timezone from the 'frame.time' column
    df['frame.time'] = df['frame.time'].str.replace(' Mountain Standard Time', '', regex=False)

    # Convert the 'frame.time' to datetime objects, coercing errors
    df['frame.time'] = pd.to_datetime(df['frame.time'], errors='coerce')

    # Drop rows where the time could not be parsed
    df = df.dropna(subset=['frame.time'])

    # Subtract the minimum time from all times to reset starting time to 0
    df['t'] = (df['frame.time'] - df['frame.time'].min()).dt.total_seconds()

    # Select the 'frame.len' as 'l' and drop all other columns
    df_final = df[['t', 'frame.len']].rename(columns={'frame.len': 'l'})

    # Set the direction for all entries
    df_final['d'] = direction

    # Save the final DataFrame to the new CSV file
    df_final.to_csv(output_file, index=False)

def process_file(task):
    input_file, output_file, direction = task
    clean_timestamps(input_file, output_file, direction)
    print(f"Converted {input_file} to {output_file}")

def ensure_dir_exists(directory):
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

def find_csv_files(directory):
    return [f for f in Path(directory).rglob('*.csv')]

def main(input_dir, output_dir, direction):
    input_dir_path = Path(input_dir)
    output_dir_path = Path(output_dir)
    ensure_dir_exists(output_dir_path)

    csv_files = find_csv_files(input_dir_path)
    tasks = []
    for csv_path in csv_files:
        relative_path = csv_path.relative_to(input_dir_path)
        output_file_path = output_dir_path / relative_path
        ensure_dir_exists(output_file_path.parent)
        tasks.append((csv_path, output_file_path, direction))

    with ProcessPoolExecutor() as executor:
        executor.map(process_file, tasks)

    print("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert timestamps in CSV files to time series data with direction, preserving directory structure.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input .csv files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the converted .csv files')
    parser.add_argument('--direction', type=int, default=1, help='Direction of traffic (1 for incoming, 0 for outgoing)')
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.direction)

# python traffic2timeSeries.py --input_dir ../data_processed/traffic_W_incoming_csv --output_dir ../data_processed/traffic_W_incoming_timeSeries_removeD
# python traffic2timeSeries.py --input_dir ../data_processed/traffic_W_outgoing_csv --output_dir ../data_processed/traffic_W_outgoing_timeSeries_removeD
# python traffic2timeSeries.py --input_dir ../data_processed/traffic_W_incoming_csv --output_dir ../data_processed/traffic_W_incoming_timeSeries --direction 1
# python traffic2timeSeries.py --input_dir ../data_processed/traffic_W_outgoing_csv --output_dir ../data_processed/traffic_W_outgoing_timeSeries --direction 0    