if test (count $argv) -ne 3
    echo "Usage: ./script_name.fish <source_folder> <destination_folder> <search_string>"
    exit 1
end

set source_folder $argv[1]
set destination_folder $argv[2]
set search_string $argv[3]

for file in (find $source_folder -name "*.log")
    if grep -q $search_string $file
        mv $file $destination_folder
    end
end
