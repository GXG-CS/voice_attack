def compare_files(file1_path, file2_path):
    """Compares two files line by line and prints line numbers where they differ."""
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        line_number = 1
        different_lines = []

        while True:
            line1 = file1.readline()
            line2 = file2.readline()

            # Stop when both files end
            if not line1 and not line2:
                break

            # Check if lines are different
            if line1 != line2:
                different_lines.append(line_number)

            line_number += 1

        return different_lines

def main():
    file1_path = '/Users/xiaoguang_guo@mines.edu/Documents/voice_attack/data_processed/vc_200/alexa/traffic_W_count_results.txt'  # Adjust this path to the first file
    file2_path = '/Users/xiaoguang_guo@mines.edu/Documents/voice_attack/data_processed/vc_200/alexa/traffic_W_filtered_count_results.txt'  # Adjust this path to the second file
    
    different_lines = compare_files(file1_path, file2_path)
    if different_lines:
        print("Differences found at lines:", different_lines)
    else:
        print("No differences found between the files.")

if __name__ == "__main__":
    main()
