import os

def remove_outliers(base_path, threshold_percentage=10, size_multiple=10):
    # Iterate through each subfolder
    for subdir, dirs, files in os.walk(base_path):
        file_sizes = []
        file_paths = []

        # Collect file sizes and paths for .pcap files
        for file in files:
            if file.endswith(".pcap"):
                file_path = os.path.join(subdir, file)
                file_size = os.path.getsize(file_path)
                file_sizes.append(file_size)
                file_paths.append(file_path)

        # Calculate the median size and determine the lower and upper size thresholds
        if file_sizes:
            median_size = sorted(file_sizes)[len(file_sizes) // 2]
            lower_threshold = (threshold_percentage / 100) * median_size
            upper_threshold = size_multiple * median_size

            # Remove files smaller than the lower threshold or larger than the upper threshold
            for file_size, file_path in zip(file_sizes, file_paths):
                if file_size < lower_threshold or file_size > upper_threshold:
                    os.remove(file_path)
                    print(f"Removed file outside the range: {file_path}")

if __name__ == "__main__":
    base_path = '../../data_processed/WiSec_unmonitored_trimmed_5'  # Change this to your directory path
    remove_outliers(base_path)
