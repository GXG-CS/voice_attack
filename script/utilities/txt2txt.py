import os

def split_lines_to_files(input_file_path):
    """
    Split each line of the input file into separate text files within a specific folder.
    Each line is saved to a different file named with a 'simple_' prefix followed by a sequential number.
    
    Args:
    - input_file_path (str): Path to the input file containing the lines to split.
    """
    # Folder where output files will be stored
    output_folder = 'text_A'
    os.makedirs(output_folder, exist_ok=True)

    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for index, line in enumerate(lines, start=1):
        output_file_path = os.path.join(output_folder, f'{index}.txt')
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(line.strip())

# Example usage:
# Replace 'path/to/your/input_file.txt' with the actual path to your input file.
split_lines_to_files('/Users/xiaoguang_guo@mines.edu/Documents/voice_attack/config/Alexa_voice_commands_200.txt')
