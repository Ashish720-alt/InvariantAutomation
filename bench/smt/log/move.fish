# Check if the log directory path and search string are provided as arguments
if test (count $argv) -lt 2
    echo "Usage: $argv[0] <log_dir> <search_string>"
    exit 1
end

# Get the log directory path and search string from the arguments
set log_dir $argv[1]
set search_string $argv[2]

# Create the target directory if it doesn't exist
set target_dir $log_dir/$search_string
mkdir -p $target_dir

# Find all log files in the log directory
for log_file in (find $log_dir -maxdepth 1 -name "*.log")
    # Check if the log file contains only one line with the search string
        if test (cat $log_file | grep $search_string)
            # Move the log file to the target directory
            mv $log_file $target_dir/
            echo "Moved $log_file to $target_dir"
        end
end