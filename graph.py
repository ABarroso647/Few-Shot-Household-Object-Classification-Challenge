import matplotlib.pyplot as plt

# main graphing function called by all the values
def graph_function(list_value, y_title, title, starting_epoch = 1):
    indices = list(range(starting_epoch, len(list_value) + starting_epoch))
    # essentially make x axis start at starting point
    plt.plot(list(range(len(list_value))), list_value,color='r')
    if len(indices) > 20:
        plt.xticks(list(range(starting_epoch, len(list_value) + starting_epoch, 5)))
    else:
        plt.xticks(list(range(len(list_value))), indices)
    plt.xlabel('Epoch #')
    plt.ylabel(y_title)
    plt.title(title)
    plt.show()