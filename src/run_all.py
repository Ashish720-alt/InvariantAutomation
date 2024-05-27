from multiprocessing import Pool
from input import Inputs, input_to_repr
from main import main
import argparse
import time

def run_task(task):
    print(f"Running task {task}")
    start = time.time()
    main(task)
    print(f"Task {task} completed. Time taken: {time.time() - start}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process pool size")
    parser.add_argument('-p', '--pool_size', type=int,
                        required=True, help='Size of the process pool')
    args = parser.parse_args()

    tasks = []
    for folder in Inputs.__dict__:
        if not folder.startswith("__"):
            subfolder = getattr(Inputs, folder)
            for inp in subfolder.__dict__:
                if not inp.startswith("__"):
                    tasks.append(input_to_repr(getattr(subfolder, inp), None, None))

    pool_size = args.pool_size

    with Pool(pool_size) as p:
        results = p.map(run_task, tasks)

    print(f"Results: {results}")
