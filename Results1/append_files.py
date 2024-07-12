import os
import sys

def append_files(directory):
    output_filename = 'appended_output.txt'
    
    with open(output_filename, 'w') as outfile:
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r') as infile:
                    outfile.write(f'Filename: {filename}\n')
                    outfile.write(infile.read())
                    outfile.write('\n\n')  # Add a newline to separate contents of different files
    print(f"All files have been appended into {output_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python append_files.py <directory>")
    else:
        directory = sys.argv[1]
        append_files(directory)
