import os

def collect_file_basenames(folders):
    basenames = []
    for folder in folders:
        for root, _, files in os.walk(folder):
            for file in files:
                name = os.path.basename(file)
                frags = name.split('.')
                basename = frags[0] + '.' + frags[1]
                basenames.append(basename)
    return basenames

# Example usage:
folders = ['./1', './2', './3', './OOM', './Timeout/']  # Replace with your folder paths
smts  = set(collect_file_basenames(folders))

log_folders = ['./log/OOM', './log/timeout-rerun/', './log/unknown/', './log/sat']
logs = set(collect_file_basenames(log_folders))

print(logs)
print(len(logs))
print(smts)
print(len(smts))

print(smts.difference(logs))
# print(smts - logs)


