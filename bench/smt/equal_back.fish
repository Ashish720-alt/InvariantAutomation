#!/usr/bin/env fish

# Check if a source folder argument is provided
if not set -q argv[1]
    echo "Usage: equal_back.fish /path/to/source/folder"
    exit 1
end

# Define the source folder from the argument
set source_folder $argv[1]

# Loop over each of the 8 folders and move files back to the source folder
for i in (seq 1 8)
    set target_folder "$source_folder/$i"
    
    # Check if the folder exists before attempting to move files
    if test -d $target_folder
        # Move all files from the current subfolder back to the source folder
        mv $target_folder/* $source_folder 2>/dev/null
        
        # Optionally, remove the subfolder after moving the files
        rmdir $target_folder
    end
end

echo "Files moved back to $source_folder!"
