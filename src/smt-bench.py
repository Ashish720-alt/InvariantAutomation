import os
import json
import subprocess
import argparse
import threading
from subprocess import check_output

task_lock = threading.Lock()
log_lock = threading.Lock()
tasks = []
timeout = None
log_file = None
z3_path = None
result_lock = threading.Lock()
result = {}



def log(string):
    log_lock.acquire()
    with open(log_file, "a") as f:
        f.write(f"{string}\n")
    log_lock.release()


def run_task():
    while True:
        task_lock.acquire()
        if not tasks:
            task_lock.release()
            return None 
        file_path = tasks.pop(0)
        task_lock.release()

        title = f"Task {file_path}"
        print(f"Running {title}")
        try:
            output = check_output([z3_path, file_path], timeout=timeout)
            print(f"File {file_path} completed successfully with {output}.")
            log(f"\"{file_path}\": \"{output}\",")
        except subprocess.TimeoutExpired:
            output = "Time out"
            print(f"File {file_path} timed out after {timeout} seconds. Skipping...")
            log(f"\"{file_path}\": \"timeout\",")
        except subprocess.CalledProcessError as e:
            output = f"Error: {e}"
            print(f"Error processing file {file_path}: {e}")
            log(f"\"{file_path}\": \"error: {e}\",")
        
        result_lock.acquire()
        result[file_path] = output
        result_lock.release()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run all tasks in parallel")
    parser.add_argument('-n', '--concurrent_workers', type=int,
                        required=True, help='The number of concurrent workers that will run the tasks')
    parser.add_argument('-t', '--timeout', type=float,
                        required=True, help='Timeout in hours')
    parser.add_argument('-l', '--log_file', type=str, required=True,
                        help='File to log')
    parser.add_argument("-z", "--z3_path", required=True,
                        help="Path to the z3 executable.")
    parser.add_argument("-i", "--input_folder", required=True,
                        help="Path to the input folder containing .smt files.")
    args = parser.parse_args()
    num_threads = args.concurrent_workers
    time_out = int(args.timeout * 60 * 60)
    log_file = args.log_file
    input_folder = args.input_folder
    z3_path = args.z3_path

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".smt"):
                file_path = os.path.join(root, file)
                tasks.append(file_path)

    # Create and start threads
    log("{")
    # threads = []
    # for _ in range(num_threads):
    #     thread = threading.Thread(target=run_task)
    #     thread.start()
    #     threads.append(thread)
    run_task()

    #    Wait for all threads to complete
    for thread in threads:
        thread.join()
    log("}")

    print("All tasks completed.")

