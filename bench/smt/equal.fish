#!/usr/bin/env fish

# Check if a source folder argument is provided
if not set -q argv[1]
    echo "Usage: equal.fish /path/to/source/folder"
    exit 1
end

# Define the source folder from the argument
set source_folder $argv[1]

# Create 8 target folders (named folder1, folder2, ..., folder8)
for i in (seq 1 8)
    set target_folder "$source_folder/$i"
    mkdir -p $target_folder
end

# Get a list of all files in the source folder (not directories)
set files (find $source_folder -maxdepth 1 -type f)

# Calculate how many files to move per folder
set total_files (count $files)
set files_per_folder (math (math $total_files / 8))

# Loop over the files and move them to one of the 8 subfolders
set folder_index 1
for file in $files
    set target_folder "$source_folder/$folder_index"
    mv $file $target_folder
    set folder_index (math $folder_index + 1)

    # If the folder index exceeds 8, reset it to 1
    if test $folder_index -gt 8
        set folder_index 1
    end
end

echo "Files moved successfully!"
