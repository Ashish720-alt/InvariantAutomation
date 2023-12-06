import matplotlib.pyplot as plt

def CostPlotter(costTimeList, s, filename='plot.png'):
    plt.figure(figsize=(20, 20))  # Set the figure size
    # ax = plt.gca()
    # ax.set_xlim([-250,20,000+250])
    # ax.set_ylim([-0-250,20,000+250])
    
    # Plotting the graph of costTimeList vs. time
    plt.plot(range(len(costTimeList)), costTimeList, marker='o', linestyle=':', label=s)

    # Labeling the axes and giving the title
    plt.xlabel('Time')
    plt.ylabel('Cost of Invariant')
    plt.title('Invariant Cost vs. Time plot')
    plt.legend()  # Show legend with label 's'

    # Add gridlines
    plt.grid(True)

    # Save the plot to a file
    plt.tight_layout()  # Adjust layout for better appearance
    plt.savefig(filename)

# # Example usage:
# integer_list = [ 5 + 5*i for i in range(100000) ]
# string_label = "z3Calls="  # Replace this string with your label
# plot_filename = "Z3Calls_2"  # File name to save the plot

# CostPlotter(integer_list, string_label, plot_filename)