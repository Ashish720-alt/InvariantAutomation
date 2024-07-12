import re

with open('appended_output.txt', 'r') as f:
    content = f.read()

with open('appended_outputSubset.txt', 'r') as f2:
    content2 = f2.read()

# Split content by '=== Contents of ... ==='
# sections = content.split('===')[1:]
sections = re.split(r"\n=== Contents of (.*\.txt) ===\n", content)[1:]

sections2 = content2.split('Filename: ')[1:]

sections2_modified = []

for A in sections2:
    # Replace the first newline character with ';;'
    modified_A = A.replace('\n', ';;', 1)
    sections2_modified.append(modified_A)

# sections2_modified now contains the modified entries


concatenated_pairs = []

# Iterate through the list with a step of 2
for i in range(0, len(sections), 2):
    if i + 1 < len(sections):  # Ensure there are always pairs to concatenate
        concatenated_pairs.append(sections[i] + ";;" + sections[i + 1])

concatenated_pairs = concatenated_pairs + sections2_modified
    
entries = []

for entry in concatenated_pairs:
    # Split the entry into filename and content based on ';;'
    filename, content = entry.split(';;', 1)
    
    # Extract total MCMC time and total Z3 time using regex
    mcmc_time = re.search(r'Total MCMC time: ([\d.]+)', content)
    z3_time = re.search(r'Total Z3 Time: ([\d.]+)', content)
    
    if mcmc_time and z3_time:
        mcmc_time = float(mcmc_time.group(1))
        z3_time = float(z3_time.group(1))
        
        entries.append((filename.strip(), mcmc_time, z3_time))

    
import matplotlib.pyplot as plt


# Extracting data
labels = [e[0] for e in entries]
n1_values = [e[1] for e in entries]
n2_values = [e[2] for e in entries]

# Creating the plot
plt.figure(figsize=(10, 6))
plt.scatter(n1_values, n2_values, marker='o', color='blue')
plt.plot([min(n1_values), max(n1_values)], [min(n1_values), max(n1_values)], color='red', linestyle='--', label='y = x line')



# Adding labels and title
plt.title('Plot of Total SA Time vs Total DS Time')
plt.xlabel('Total SA Time (seconds)')
plt.ylabel('Total DS Time (seconds)')

# Adding text labels for each point
for label, n1, n2 in zip(labels, n1_values, n2_values):
    plt.text(n1, n2, "", fontsize=8, ha='right', va='bottom')

# Display the plot
plt.grid(True)
plt.tight_layout()
plt.show()
