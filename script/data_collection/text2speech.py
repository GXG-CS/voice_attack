import argparse
import torch
import os
from TTS.api import TTS
import torch.multiprocessing as mp
import time

def process_text_file(args):
    # Unpack arguments
    model_name, text_file_path, audio_file_path = args
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts = TTS(model_name=model_name).to(device)
        
        with open(text_file_path, 'r') as f:
            text = f.read()

        # Text-to-speech conversion
        tts.tts_to_file(text=text, file_path=audio_file_path)
        print(f"Processed {os.path.basename(text_file_path)}")
    except Exception as e:
        print(f"Error processing {os.path.basename(text_file_path)}: {str(e)}")

def parallel_processing(tasks, num_processes):
    start_time = time.time()
    with mp.Pool(processes=num_processes) as pool:
        pool.map(process_text_file, tasks)
    end_time = time.time()
    print(f"Parallel processing time with {num_processes} processes: {end_time - start_time} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process text files to generate audio.')
    parser.add_argument('--model_name', type=str, required=True, help='TTS model name')
    parser.add_argument('--text_files_directory', type=str, required=True, help='Directory of text files')
    parser.add_argument('--base_audio_files_directory', type=str, required=True, help='Base directory for output audio files')
    parser.add_argument('--num_processes', type=int, default=min(mp.cpu_count(), 10), help='Number of parallel processes')

    args = parser.parse_args()

    # Set 'spawn' start method
    mp.set_start_method('spawn', force=True)
    
    tasks = []
    os.makedirs(args.base_audio_files_directory, exist_ok=True)

    for text_file in os.listdir(args.text_files_directory):
        if text_file.endswith('.txt'):
            text_file_path = os.path.join(args.text_files_directory, text_file)
            audio_file_name = f"{text_file.replace('.txt', '')}.wav"
            audio_file_path = os.path.join(args.base_audio_files_directory, audio_file_name)
            tasks.append((args.model_name, text_file_path, audio_file_path))

    parallel_processing(tasks, args.num_processes)


# python text2speech.py --model_name "tts_models/en/ljspeech/tacotron2-DCA" --text_files_directory "../../raw_data/dilawer11/simple_100_alexa/text_A" --base_audio_files_directory "../../raw_data/dilawer11/simple_100_alexa/audioPlay_A" 
