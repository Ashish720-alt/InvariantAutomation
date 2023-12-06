import matplotlib.pyplot as plt

def CostPlotter(costTimeLists, threads, filename='plot.png'):
    plt.figure(figsize=(20, 20))  # Set the figure size
    # ax = plt.gca()
    # ax.set_xlim([-250,20,000+250])
    # ax.set_ylim([-0-250,20,000+250])
    
    # Plotting the graph of costTimeList vs. time    
    winners = "Success:"
    for i in range(threads):
        costTimeList = costTimeLists[i]
        lbl = "Thread" + str(i)
        
        if (costTimeList[-1] == 0):
            winners = winners + "T" + str(i)
        plt.plot(range(1, 1+len(costTimeList)), costTimeList,  linestyle='-', label=lbl) 

    # Labeling the axes and giving the title
    plt.xlabel('Time')
    plt.ylabel('Cost of Invariant')
    plt.title('Invariant Cost vs. Time plot' + " (" + winners + ")" )
    plt.legend()  # Show legend with label 's'

    # Add gridlines
    plt.grid(True)

    # Save the plot to a file
    plt.tight_layout()  # Adjust layout for better appearance
    plt.savefig(filename)

# Example usage:
# integer_list1 = [ 5 + 5*i for i in range(100000) ]
# integer_list2 = [ 10 + 10*i for i in range(100000) ]
# integer_list3 = [ 15 + 15*i for i in range(100000) ]
# integer_list4 = [ 100000 - 5*i for i in range(100000) ]

# costlists = [ integer_list1, integer_list2, integer_list3, integer_list4 ] 
# plot_filename = "Z3Calls_2"  # File name to save the plot

# CostPlotter(costlists, 4, plot_filename)