import logging

# Configure logging at the very start
log_file_path = "process_pcap.log"
logging.basicConfig(filename=log_file_path, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S')

# Test logging
logging.info("This is a test log entry.")
