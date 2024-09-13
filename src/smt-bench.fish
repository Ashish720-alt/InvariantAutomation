# Check if the folder path and log directory path are provided as arguments
if test (count $argv) -lt 3
    echo "Usage: $argv[0] <folder_path> <log_dir> <timeout>"
    exit 1
end

# Get the folder path from the argument
set folder_path $argv[1]
# set folder_path ../bench/smt
set log_path $argv[2]
# set timeout
set timeout $argv[3]

# rm $log_path/*.log

echo "Start"

# Find all .smt files and run a command with a timeout of 4 hours (14400 seconds), saving output to separate log files
for file in (find $folder_path -name "*.smt")
  # Extract the base name of the file (without the directory)
  set base_name (basename $file)

  # Define the log file path for this specific file
  set log_file $log_path/$base_name.log

  # Check if the log file already exists
  if test -e $log_file
      echo "Log file already exists for $file. Skipping."
      continue
  end

  # Print the current time stamp
  echo (date)  
  # Run the command with a timeout and redirect output to the log file
  echo "Processing $file"
  timeout $timeout ../../z3/build/z3 $file >> $log_file 2>&1
  echo "$file Done"
end
