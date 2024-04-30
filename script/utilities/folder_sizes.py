import os

def get_folder_size(start_path):
    """Calculate the total size of all files in the folder."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def print_smallest_subfolder_sizes(total_folder):
    """Print the sizes of the top 10 smallest subfolders within the total folder."""
    sizes = []
    for subdir in os.listdir(total_folder):
        subdir_path = os.path.join(total_folder, subdir)
        if os.path.isdir(subdir_path):  # Ensure it's a directory
            size = get_folder_size(subdir_path)
            sizes.append((subdir, size))

    # Sort the list of tuples by size (second item in tuple) and select the top 10 smallest
    smallest_subfolders = sorted(sizes, key=lambda x: x[1])[:10]

    for subdir, size in smallest_subfolders:
        print(f"Size of '{subdir}': {size} bytes")

def main():
    # Hardcoded path to the total folder
    total_folder = "/Users/xiaoguang_guo@mines.edu/Documents/voice_attack/data_processed/vc_200/alexa/total"
    print_smallest_subfolder_sizes(total_folder)

if __name__ == "__main__":
    main()


