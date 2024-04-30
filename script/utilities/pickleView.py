import pickle

def view_pkl_content(file_path, num_elements=10):
    # Load the content of the pickle file
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    # Determine the number of elements in the data
    data_length = len(data) if hasattr(data, '__len__') else 'Unknown'

    # Print the type of the loaded data and its length
    print(f"Type of data: {type(data)}")
    print(f"Number of elements: {data_length}")

    # Depending on the data type, print a portion of the content
    if isinstance(data, list):
        # If it's a list, print the first few elements
        for i, element in enumerate(data[:num_elements]):
            print(f"Element {i}: {element}")
    elif isinstance(data, dict):
        # If it's a dictionary, print the first few key-value pairs
        for i, (key, value) in enumerate(list(data.items())[:num_elements]):
            print(f"Key: {key}, Value: {value}")
    else:
        # For other data types, print the data directly
        print(data)

# Example usage
if __name__ == "__main__":
    file_path = '../data_processed/traffic_W_outgoing_timeSeries_removeD_pickles/simple_1.pkl'  # Replace with actual file path
    # file_path = 'features/traffic_W_outgoing_features/features_1_tts_models_en_ljspeech_tacotron2-DCA.pkl'
    view_pkl_content(file_path)
