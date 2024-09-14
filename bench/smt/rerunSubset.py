import os
import subprocess

# Folder paths
memerr_folder = 'sat' #or 'MemErr'
bench_folder = 'bench'

# Timeout in seconds
timeout = 10

# Iterate over all files in the MemErr folder
for filename in os.listdir(memerr_folder):
    if filename.endswith('.smt.log'):
        # Derive the corresponding .smt file in the bench folder
        smt_filename = filename.replace('.smt.log', '.smt')
        smt_filepath = os.path.join(bench_folder, smt_filename)
        
        # Check if the corresponding .smt file exists in the bench folder
        if os.path.exists(smt_filepath):
            # Prepare the z3 command
            command = ['z3', smt_filepath]

            try:
                # Run the command with a timeout
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Wait for process to complete or timeout
                stdout, stderr = process.communicate(timeout=timeout)
                
                # Display only the z3 output
                print(stdout.decode())

            except subprocess.TimeoutExpired:
                process.kill()