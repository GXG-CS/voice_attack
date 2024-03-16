import os

def remove_macos_metadata_files(directory):
    """
    Remove all '._' prefixed files that are created by macOS systems when copying files to 
    non-macOS formatted drives. This function will recursively search the given directory 
    and all its subdirectories for any such files and remove them.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith('._'):
                # Construct the full path to the file
                path_to_file = os.path.join(root, file)
                # Remove the file
                os.remove(path_to_file)
                # Print out the path of the removed file for confirmation
                print(f"Removed {path_to_file}")

# Replace with the path to your top-level "total folder"
total_folder_path = "../../raw_data/dilawer11/simple_100_alexa/traffic_W_raw/"
remove_macos_metadata_files(total_folder_path)
