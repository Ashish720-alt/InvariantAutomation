import os
import subprocess

# Path to the directory containing the .smt files
bench_dir = "./bench"

# Check if the directory exists
conv = 0
if os.path.exists(bench_dir):
    # Loop through each file in the directory
    for file_name in os.listdir(bench_dir):
        if file_name.endswith(".smt"):
            file_path = os.path.join(bench_dir, file_name)
            try:
                # Run the Z3 command with a timeout of 10 seconds
                result = subprocess.run(["z3", file_path], capture_output=True, text=True, timeout=10)
                print(result.stdout)  # Print only what Z3 outputs
                conv = conv + 1
            except subprocess.TimeoutExpired:
                print(f"Timeout: z3 took too long for {file_name}")
    
    print("Converged benchmarks = ",conv)