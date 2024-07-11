from input import Inputs, input_to_repr
import os
import subprocess
from main import main
import threading
import multiprocessing
import time
import argparse

task_lock = threading.Lock()
log_lock = threading.Lock()
tasks = []
time_out = None
repeat = None
log_file = None


def log(string):
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{t} {string}")
    log_lock.acquire()
    with open(log_file, "a") as f:
        f.write(f"{t} {string}\n")
    log_lock.release()


def main_wrapper(folder, name):
    input = input_to_repr(
        getattr(getattr(Inputs, folder), name), None, None, None)
    return main(f"{folder}.{name}", input)


def run_task():
    while True:
        task_lock.acquire()
        if not tasks:
            task_lock.release()
            break
        task_name = tasks.pop(0)
        task_lock.release()
        
        for run in range(repeat):
            log(f"Running {title}")
            command = f"timeout {time_out} python main.py -i {task_name[0]}.{task_name[1]}"
            exitcode = os.system(command)
            title = f"Task {task_name[0]}.{task_name[1]} Round {run}"
            try:
                if exitcode == 31744:
                    log(f"{title} timed out")
                elif exitcode == 0:
                    log(f"{title} completed successfully")
                else:
                    log(f"{title} failed with exitcode {exitcode}")
            except Exception as e:
                log(f"Error running {title}: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run all tasks in parallel")
    parser.add_argument('-n', '--concurrent_workers', type=int,
                        required=True, help='The number of concurrent workers that will run the tasks')
    parser.add_argument('-t', '--timeout', type=float,
                        required=True, help='Timeout in hours')
    parser.add_argument('-r', '--repeat', type=int, default=1,
                        help='Number of times to repeat each task')
    parser.add_argument('-l', '--log_file', type=str, required=True,
                        help='File to log')

    args = parser.parse_args()
    num_threads = args.concurrent_workers
    time_out = int(args.timeout * 60 * 60)
    repeat = args.repeat
    log_file = args.log_file

    for folder in Inputs.__dict__:
        if not folder.startswith("__"):
            subfolder = getattr(Inputs, folder)
            for inp in subfolder.__dict__:
                if not inp.startswith("__"):
                    tasks.append((f"{folder}", f"{inp}"))

    # Create and start threads
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(target=run_task)
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    log("All tasks completed.")
