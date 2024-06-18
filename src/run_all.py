from multiprocessing import Pool
from input import Inputs, input_to_repr
from main import main
import argparse
import time
from pebble import ProcessPool
from concurrent.futures import TimeoutError, CancelledError


def run_task(task, repeat):
    print(f"Running Task {task} Round {repeat}")
    start = time.time()
    main(task)
    # time.sleep(task)
    print(
        f"Task {task} Round {repeat} completed. Time taken: {time.time() - start}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run all tasks in parallel")
    parser.add_argument('-p', '--pool_size', type=int,
                        required=True, help='Size of the process pool')
    parser.add_argument('-t', '--timeout', type=float,
                        required=True, help='Timeout in hours')
    parser.add_argument('-r', '--repeat', type=int, default=1,
                        help='Number of times to repeat the task')
    args = parser.parse_args()
    pool_size = args.pool_size
    time_out = int(args.timeout * 60 * 60)
    repeat = args.repeat

    tasks = []
    for folder in Inputs.__dict__:
        if not folder.startswith("__"):
            subfolder = getattr(Inputs, folder)
            for inp in subfolder.__dict__:
                if not inp.startswith("__"):
                    task = input_to_repr(
                        getattr(subfolder, inp), None, None, None)
                    tasks.extend([(task, i) for i in range(repeat)])
    # for i in [3*i for i in range(5)]:
    #     tasks.extend([(i, r) for r in range(repeat)])

    results = []
    with ProcessPool(max_workers=pool_size) as pool:

        futures = [
            pool.schedule(run_task, [task, repeat], timeout=time_out) for (task, repeat) in tasks]
        pool.close()
        pool.join()

        for ((task, repeat), future) in list(zip(tasks, futures)):
            try:
                # Get the result of the task
                ret = future.result()
                results.append(f"Task {task} Run {repeat} succeeded with result {ret}")
            except TimeoutError:  # Catch the TimeoutError
                results.append(f"Task {task} Run {repeat} timed out.")
            except CancelledError:
                results.append(f"Task {task} Run {repeat} cancelled.")
            except Exception:
                results.append(f"Task {task} Run {repeat} failed.")

    for result in results:
        print(result)
