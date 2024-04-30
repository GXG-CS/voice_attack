import os

def is_numeric(s):
    """Check if the string is numeric."""
    try:
        int(s)
        return True
    except ValueError:
        return False

def remove_ds_store(directory):
    """Recursively removes all .DS_Store files from the specified directory and its subdirectories."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == '.DS_Store':
                os.remove(os.path.join(root, file))
                print(f"Removed .DS_Store from {root}")

def count_files_in_subfolders(total_folder, output_file):
    """Prints the number of files in each subfolder within the total folder, sorted by the number of files."""
    total_files_count = 0
    subfolders = os.listdir(total_folder)

    # Filter out non-numeric subfolder names and sort numerically
    subfolders = [s for s in subfolders if is_numeric(s)]
    subfolders.sort(key=int)

    with open(output_file, 'w') as file:
        subfolder_file_counts = {}
        for subdir in subfolders:
            subdir_path = os.path.join(total_folder, subdir)
            if os.path.isdir(subdir_path):  # Check if it is a directory
                files = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
                file_count = len(files)
                subfolder_file_counts[subdir] = file_count
                total_files_count += file_count

        # Write subfolder counts sorted by the number of files
        sorted_subfolders = sorted(subfolder_file_counts.items(), key=lambda x: x[1])
        for subdir, file_count in sorted_subfolders:
            file.write(f"'{subdir}' has {file_count} files.\n")

        file.write(f"Total number of files in '{total_folder}': {total_files_count}\n")

def main():
    # total_folder = '/Users/xiaoguang_guo@mines.edu/Documents/voice_attack_data/data_processed/vc_200/alexa/trimmed_1s/outgoing_trimmed'
    total_folder = '/Users/xiaoguang_guo@mines.edu/Documents/voice_attack_data/data_processed/vc_200/google/no_trim/raw'

    output_file = "/Users/xiaoguang_guo@mines.edu/Documents/voice_attack_data/data_processed/vc_200/google/no_trim/raw_file_counts.txt"
    
    # Remove .DS_Store files before counting
    remove_ds_store(total_folder)
    
    # Count files in subfolders
    count_files_in_subfolders(total_folder, output_file)

if __name__ == "__main__":
    main()
