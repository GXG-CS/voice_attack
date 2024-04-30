import pandas as pd
from pathlib import Path
import pickle
import argparse
from concurrent.futures import ProcessPoolExecutor

def load_traffic_time_series_data(file_path):
    print(f"Loading file: {file_path.name}")  # Print the current file name
    traffic_df = pd.read_csv(file_path)
    # Convert DataFrame to a list of tuples (t, l, d)
    return list(traffic_df.itertuples(index=False, name=None))

def process_directory(directory_path):
    # Sort files by the numerical value in their filename
    file_paths = sorted(Path(directory_path).glob('*.csv'), key=lambda path: int(path.stem.split('_')[0]))
    with ProcessPoolExecutor() as executor:
        traffic_data_as_tuples = list(executor.map(load_traffic_time_series_data, file_paths))
    return traffic_data_as_tuples

def save_data_to_pickle(data, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {output_file}")

def main(traffic_data_root_directory, output_directory):
    # Ensure the output directory exists
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    
    # Process each subdirectory within the root directory
    for subdirectory in Path(traffic_data_root_directory).iterdir():
        if subdirectory.is_dir():
            print(f"Processing directory: {subdirectory.name}")
            traffic_data_as_tuples = process_directory(subdirectory)
            output_file = Path(output_directory, f"{subdirectory.name}.pkl")
            save_data_to_pickle(traffic_data_as_tuples, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process traffic time series data directories and save each as a pickle file.')
    parser.add_argument('--traffic_data_root_directory', type=str, required=True, help='Root directory containing subdirectories of traffic time series CSV files')
    parser.add_argument('--output_directory', type=str, required=True, help='Directory to save the pickle files')
    args = parser.parse_args()

    main(args.traffic_data_root_directory, args.output_directory)

# python traffic_preprocessor2.py --traffic_data_root_directory ../data_processed/traffic_W_outgoing_timeSeries_removeD --output_directory ../data_processed/traffic_W_outgoing_timeSeries_removeD_pickles
# python traffic_preprocessor2.py --traffic_data_root_directory ../data_processed/traffic_W_incoming_timeSeries_removeD --output_directory ../data_processed/traffic_W_incoming_timeSeries_removeD_pickles
# python traffic_preprocessor2.py --traffic_data_root_directory ../data_processed/traffic_W_outgoing_timeSeries --output_directory ../data_processed/traffic_W_outgoing_timeSeries_pickles
# python traffic_preprocessor2.py --traffic_data_root_directory ../data_processed/traffic_W_incoming_timeSeries --output_directory ../data_processed/traffic_W_incoming_timeSeries_pickles        