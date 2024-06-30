import os
import subprocess
import argparse

def process_files(z3_path, input_folder, timeout):
    # Find all .smt files in the input folder
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".smt"):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                
                # Run the z3 command on the file with a timeout
                try:
                    subprocess.run([z3_path, file_path], check=True, timeout=timeout)
                except subprocess.TimeoutExpired:
                    print(f"File {file_path} timed out after {timeout} seconds. Skipping...")
                except subprocess.CalledProcessError as e:
                    print(f"Error processing file {file_path}: {e}")
                
                # Wait for a keystroke to continue
                # input("Press any key to continue to the next file...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process .smt files with z3 command.")
    parser.add_argument("-z", "--z3_path", required=True, help="Path to the z3 executable.")
    parser.add_argument("-t", "--timeout", type=int, default=10, help="Timeout for each file processing in seconds (default: 10).")
    parser.add_argument("-i", "--input_folder", required=True, help="Path to the input folder containing .smt files.")
    args = parser.parse_args()
    process_files(args.z3_path, args.input_folder, args.timeout)