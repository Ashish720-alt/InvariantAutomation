# Check if both log folder and SMT folder paths are provided as arguments
if test (count $argv) -lt 2
    echo "Usage: $argv[0] <log_folder> <smt_folder>"
    exit 1
end

# Get the log folder and SMT folder paths from the arguments
set log_folder $argv[1]
set smt_folder $argv[2]

mkdir -p $smt_folder/$log_folder
# Iterate over each log file in the log folder
for log_file in (find $log_folder -type f -name "*.log")
    # Extract the base name of the log file (without the directory and extension)
    set base_name (basename $log_file .smt.log)
    echo $base_name
    
    # Find the SMT file with the same base name recursively in the SMT folder
    set smt_file (find $smt_folder -type f -name "$base_name.smt")
    
    # Check if the SMT file exists
    if test -n "$smt_file"
        echo "Found SMT file: $smt_file"
        
        # Run the specified command on the SMT file
        # Replace 'your_command' with the actual command you want to run
        mv $smt_file $smt_folder/$log_folder/$base_name.smt
    else
        echo "No SMT file found for log file: $log_file"
    end
end