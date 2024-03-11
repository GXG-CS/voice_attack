import torch
import os
from TTS.api import TTS
import torch.multiprocessing as mp
import time

def process_text_file(args):
    model_name, text_file_path, audio_file_path = args
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts = TTS(model_name).to(device)
        
        with open(text_file_path, 'r') as f:
            text = f.read()

        tts.tts_to_file(text=text, file_path=audio_file_path)
        print(f"Processed {text_file_path} using model {model_name}")
    except Exception as e:
        print(f"Error processing {text_file_path} with model {model_name}: {str(e)}")

def sequential_processing(tasks):
    start_time = time.time()
    for task in tasks:
        process_text_file(task)
    end_time = time.time()
    print(f"Sequential processing time: {end_time - start_time} seconds")

def parallel_processing(tasks, num_processes):
    start_time = time.time()
    pool = mp.Pool(processes=num_processes)
    pool.map(process_text_file, tasks)
    pool.close()
    pool.join()
    end_time = time.time()
    print(f"Parallel processing time with {num_processes} processes: {end_time - start_time} seconds")

if __name__ == "__main__":
    # Ensure to use 'spawn' start method
    mp.set_start_method('spawn', force=True)
    
    text_files_directory = "text_A"
    base_audio_files_directory = "audioPlay_A"
    models = ["tts_models/en/ljspeech/tacotron2-DCA"]

    tasks = []
    for model_name in models:
        safe_model_name = model_name.replace("/", "_").replace(":", "_")
        audio_files_directory = os.path.join(base_audio_files_directory, safe_model_name)
        os.makedirs(audio_files_directory, exist_ok=True)

        for text_file in os.listdir(text_files_directory):
            if text_file.endswith('.txt'):
                text_file_path = os.path.join(text_files_directory, text_file)
                audio_file_name = f"{text_file.replace('.txt', '')}_{safe_model_name}.wav"
                audio_file_path = os.path.join(audio_files_directory, audio_file_name)
                tasks.append((model_name, text_file_path, audio_file_path))

    # Optionally, uncomment to run sequential processing for comparison
    # sequential_processing(tasks)

    # Parallel processing
    num_processes = min(mp.cpu_count(), 10)  
    parallel_processing(tasks, num_processes)
