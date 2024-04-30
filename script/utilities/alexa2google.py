def replace_alexa_with_google(input_file_path, output_file_path):
    with open(input_file_path, 'r') as file:
        lines = file.readlines()
    
    updated_lines = [line.replace("Alexa", "Okay, Google") for line in lines]
    
    with open(output_file_path, 'w') as file:
        file.writelines(updated_lines)

# Example usage
input_file_path = '/Users/xiaoguang_guo@mines.edu/Documents/voice_attack/config/Alexa_voice_commands_200.txt'  # Replace with the actual path
output_file_path = 'updated_voice_commands.txt'  # Replace with the desired output path

replace_alexa_with_google(input_file_path, output_file_path)
