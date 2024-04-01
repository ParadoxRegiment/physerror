#       #####################################       #  
#        Module written by Alexander Ritzie         #
#       Originally created for BPHYS 312 AU23       #
#       at University of Washington - Bothell       #
#                   GitHub Repo:                    #
#    https://github.com/ParadoxRegiment/BPHYS231    #
#                                                   #
#      Please direct any questions to my email:     #
#            alexander.ritzie@gmail.com             #
#       #####################################       #

import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pandas as pd
from dataclasses import dataclass, field
import seaborn as sns

@dataclass
class Data():
    """ An initializer and container dataclass that is initialized and reused
        by the user and other classes, as well as their methods. There are many 
        attributes that calculate common statistical errors, constants, and 
        general error propagation methods, and one class Method to find, document, 
        and delete any data points that exist outside the standard 2 * sigma outlier
        "limit".
        
        Attributes:
            delta : float
                []
            A : float
                []
            B : float
                []
            x_mean : float
                [] 
            x_best : float
                []
            y_mean : float
                []
            y_best : float
                [] 
            sigma_y : float
                []
            sigma_A : float
                []
            sigma_B : float
                []
            sigma_x : float
                []
            sigma_x_best : float
                []
            sigma_y_best : float
                []
            sigma_x_mean : float
                []
            sigma_y_mean : float
                []
            sigma_frac : float
                []
    
    #### Will update this soon-ish ####
    """
    user_x_data : np.ndarray
    user_y_data : np.ndarray
    _x_data : np.ndarray = field(init=False)
    _y_data : np.ndarray = field(init=False)
    _df : pd.DataFrame = field(init=False)
    _colname1 : str = field(init=False)
    _colname2 : str = field(init=False)
    
    def __post_init__(cls):
        cls._x_data, cls._y_data, cls._df, \
            cls._colname1, cls._colname2 = cls._initdata(cls.user_x_data, cls.user_y_data)
        cls._initcalcs()

    def _initcalcs(cls):
        # Checks that cls._x_data is not an empty array. If it is, it does not calculate
        # any of the attributes inside the if statement
        if len(cls._x_data) != 0:
            N = len(cls._x_data)
            cls.delta = N*sum(cls._x_data**2) - (sum(cls._x_data))**2
            cls.A = (((sum(cls._x_data**2)*sum(cls._y_data)) - (sum(cls._x_data)*sum(cls._x_data*cls._y_data)))/cls.delta)
            cls.B = (N * sum(cls._x_data*cls._y_data) - (sum(cls._x_data)*sum(cls._y_data))) / cls.delta
            cls.x_mean = abs(np.mean(cls._x_data))
            cls.x_best = sum(cls._x_data)/N
            cls.y_mean = abs(np.mean(cls._y_data))
            cls.y_best = sum(cls._y_data)/N
            cls.sigma_y = np.sqrt((1/(N - 2))*sum((cls._y_data - cls.A - (cls.B * cls._x_data))**2))
            cls.sigma_A = cls.sigma_y*np.sqrt(sum(cls._x_data**2)/cls.delta)
            cls.sigma_B = cls.sigma_y*np.sqrt(N/cls.delta)
            cls.sigma_x = np.sqrt(sum((cls._x_data - cls.x_mean)**2)/(N - 1))
            cls.sigma_x_best = np.sqrt((1/(N - 1))*sum((cls._x_data - cls.x_mean)**2))
            cls.sigma_y_best = np.sqrt((1/(N - 1))*sum((cls._y_data - cls.y_mean)**2))
            cls.sigma_x_mean = cls.x_mean/np.sqrt(N)
            cls.sigma_y_mean = cls.y_mean/np.sqrt(N)
            cls.sigma_frac = 1/np.sqrt(2 * (N - 1))
    
    # Initializes and returns the data that will be reused
    def _initdata(cls, xdata = np.arange(5) + 1, ydata = np.arange(5) + 1):
        """ Callable but largely useless if done so. Used to read in a csv if
            the user so wishes, store data and their user-inputed names, then
            returns that data back to the line where it was called. It is only
            used inside the __init__ function of this class.
            
            Parameters:
                xdata : np.ndarray
                    The given x data. As it is used, it is the "user_x_data"
                    passed in at the time of initialization.
                
                ydata : np.ndarray
                    The given y data. As it is used, it is the "user_y_data"
                    pass in at the time of initialization.
        
            Returns:
                x_data -> np.ndarray
                    The x_data array created from either the passed-in
                    user_x_data or the csv file that was read in by the user.
                
                y_data -> np.ndarray
                    The y_data array created from either the passed-in
                    user_y_data or the csv file that was read in by the user.
                
                datafile -> pd.DataFrame
                    The pandas DataFrame created from either the passed-in
                    user_x_data and user_y_data or the csv file that was
                    read in by the user.
                
                colname1 -> str
                    The name entered by the user for x_data.
                
                colname2 -> str
                    The name entered by the user for y_data.
        """

        readcsv = input("Would you like to read in a CSV? Y/N\n")
        
        # Calls for the csvreader function if the user enters a yes(-adjecent) input
        if readcsv.lower() == "y" or readcsv.lower() == "yes":
            x_data, y_data = csvreader()
            
        # Assumes the user's passed in arguments are the desired data and passes them
        # into new variables                                                                             
        elif readcsv.lower() == "n" or readcsv.lower() == "no":
            x_data = xdata
            y_data = ydata
            pass
        else:
            print("Unknown input, please restart.")
            exit()
        
        # Stacks the arrays to be turned into a pandas DataFrame
        temparray = np.stack((x_data, y_data))
        colname1 = input("Please type the first data set's name: ")
        colname2 = input("Please type the second data set's name: ")
        
        # Creates a DataFrame using the transpose of the array stack for data and the
        # user input data names for column names
        datafile = pd.DataFrame(np.transpose(temparray), columns = [colname1, colname2])
        
        # Renames the df index name to Trial
        datafile.index.name = 'Trial'
        
        # Iterates the index count by one for readability
        datafile.index += 1
        
        # Deletes temparray as it will not be used again
        del temparray
        
        # Prints out the df for the user to see
        print(datafile)
        
        # Returns the given variables into the Class' cls-variables
        return x_data, y_data, datafile, colname1, colname2
    
    
    ##### Will do docstring documentation later #####
    def outlier(cls):
        """ A method that creates two empty arrays then searches the
            cls.x_data and cls.y_data arrays for values that are outside
            the standard 2 * sigma outlier "limit".

            Returns:
                x_outliers -> np.ndarray
                    An array that contains either the outliers that were found
                    in the user's x_data, or a string stating no outliers were
                    found.
                
                y_outliers -> np.ndarray
                    An array that contains either the outliers that were found
                    in the user's y_data, or a string stating no outliers were
                    found.
        """
        
        # New x_data and y_data variables for ease of use
        x_data = cls._x_data
        y_data = cls._y_data
        
        # Immediately exits the program if the x_data or y_data are somehow empty arrays
        if np.size(x_data) == 0 or np.size(y_data) == 0:
            exit()
        
        # Outlier variables to be used for later
        x_outliers = np.zeros(len(x_data))
        y_outliers = np.zeros(len(y_data))
        
        # Iterater variables for x_data, x_outliers, y_data, and y_outliers respectively
        i = 0                                               
        j = 0
        k = 0
        l = 0
        
        # Loops through the x_data array to check for outliers
        for row in x_data:
            
            # Checks if the row value is greater than the mean + 2*sigma
            if row > (cls.x_mean + 2 * cls.sigma_x):      
                
                # If above is true, inserts the row value into the j cell of x_outliers
                x_outliers[j] = int(row)   
                
                # Iterates j by one                 
                j += 1        
                
                # Deletes the outlier cell from x_data                              
                x_data = np.delete(x_data, i)               
                
            # Checks if the row value is less than the mean - 2*sigma
            elif row < (cls.x_mean - 2*cls.sigma_x):    
                
                # If above is true, inserts the row value into the j cell of x_outliers
                x_outliers[j] = int(row)                    
                
                # Iterates j by one
                j += 1                                      
                
                # Deletes the outlier cell from x_data
                x_data = np.delete(x_data, i)               
                
            # Iterates i by one
            i += 1                                          
        
        # Loops through the y_data array to check for outliers
        for row in y_data:                                  
            
            # Checks if the row value is greater than the mean + 2*sigma
            if row > (cls.y_mean + 2*cls.sigma_y):      
                
                # If above is true, inserts the row value into the l cell of y_outliers
                y_outliers[l] = int(row)                    
                
                # Iterates l by one
                l += 1                                      
                
                # Deletes the outlier cell from y_data
                y_data = np.delete(y_data, k)               
            
            # Checks if the row value is less than the mean - 2*sigma
            elif row < (cls.y_mean - 2*cls.sigma_y):    
                
                # If above is true, inserts the row value into the l cell of y_outliers
                y_outliers[l] = int(row)                    
                
                # Iterates j by one
                l += 1                                      
                
                # Deletes the outlier cell from y_data
                y_data = np.delete(y_data, k)
                
            # Iterates k by one
            k += 1                                          
            
        # Resizes the x_outliers array to the size of j to remove redundant zeroes
        x_outliers.resize(j)                                
        
        # Resizes the y_outliers array to the size of l to remove redundant zeroes
        y_outliers.resize(l)                                
        
        # Checks if there were no outliers in x_data
        if np.size(x_outliers) == 0:                        
            
            # If above is true, reinitializes x_outliers to the given string
            x_outliers = 'No outliers in x data'            
        
        # Checks if there were no outliers in y_data
        if np.size(y_outliers) == 0:                        
            
            # If above is true, reinitializes y_outliers to the given string
            y_outliers = 'No outliers in y data'            

        return x_outliers, y_outliers

class Graphs:
    """ Allows the user to create various graphs from the user_data
        pulled from Data. There is no __init__ method for this Class.
    """
    
    def linreg(user_data : Data, gtitle = "Graph"):
        """ Uses the given x_data and y_data arrays to create a linear
            regression plot.
            
            Parameters:
                user_data : Data
                    Requires the user to pass in an instance of
                    Data to make use of the user's data.
                    
                gtitle : str [optional]
                    The desired graph title. Defaults to "Graph".
            
            Returns:
                plt.show()
                    Opens an external window that shows the linear
                    regression plot of the given data.
        """
        
        # New x_data and y_data for ease of use
        x_data = user_data._x_data                                
        y_data = user_data._y_data
        
        # Sets the figure's title to the default (or passed in) graph title
        plt.title(gtitle, fontsize = 11)                        
        
        # Sets the figure data to x_data and y_data, colored orange
        plt.plot(x_data, y_data, 'o')                           
        
        # Sets the figure's xlabel to the user's entered x_data name
        plt.xlabel(user_data._colname1, fontsize = 11)            
        
        # Sets the figure's xlabel to the user's entered y_data name
        plt.ylabel(user_data._colname2, fontsize = 11)            
        
        # Adds the linear regression line to the plot
        plt.plot(x_data, user_data.A + user_data.B * x_data)      
        
        # Displays the linear regression plot
        plt.show()                                              
    
    def errbargraph(user_data : Data, gtitle = "Graph"):
        """ Uses the given dataframe built from x_data and y_data during
            initalization to create an error bar plot, making use of
            the sigma_x value as the constant error.
            
            Parameters:
                user_data : Data
                    Requires the user to pass in an instance of
                    Data to make use of the user's data.
                    
                gtitle : str [optional]
                    The desired graph title. Defaults to "Graph".
            
            Returns:
                plt.show()
                    Opens an external window that displays the
                    error bar graph.
        """
        
        # New df for ease of use
        df = user_data._df                                                                    
        
        # Creates an errorbar graph out of the given df data. Sets x to the y_data and y
        # to the x_data on an assumption that the y_data is the independent variable
        # yerr is set to the user_data's sigma_x
        df.plot(x = user_data._colname2, y = user_data._colname1,                               
                xlabel = user_data._colname2, ylabel = user_data._colname1, title = gtitle,
                linestyle = "", marker = ".", yerr = user_data.sigma_x,
                capthick = 1, ecolor = "red", linewidth = 1)
        plt.show()
    
    def datahist(user_data : Data, gtitle = "Graph"):
        """ Uses the given dataframe built from x_data and y_data during
            initalization to create one or two histograms. There is also
            the option to turn the graphs into standard distribution
            graphs (currently a WIP).
        
            Parameters:
                user_data : Data
                    Requires the user to pass in an instance of
                    Data to make use of the user's data.
                    
                gtitle : str [optional]
                    The desired graph title. Defaults to "Graph".
            
            Returns:
                plt.show()
                    Opens an external window that displays the
                    histogram(s).
        """
        # New df for ease of use
        datafile = user_data._df                                                                          
        
        # Initialized to the df's columns array for future use
        columns = datafile.columns                                                                      
        
        stdcheck = input("Will this a standard distribution graph? Y/N ")
        
        # Internal function used to determine what type of histogram graph will be created
        def stdcheckfunc():
            
            # Checks if the user entered a yes(-adjecent) input
            if stdcheck.lower() == "y" or stdcheck.lower() == "yes":
                
                # Pulls the minimum and maximum for the x-axis from data later down
                xmin, xmax = plt.xlim()                                                                 
                
                # Creates an x-axis array of 100 values
                x = np.linspace(xmin, xmax, 100)                                                        
                
                # Calls the scipy.stats.pdf method with x as the data, user_data x_mean as the mean,
                # and user_data sigma_x as the scale
                p = stats.norm.pdf(x, user_data.x_mean, user_data.sigma_x)                                
                
                # Creates a graph with x on the x-axis and p on the y-axis
                plt.plot(x, p, 'k', linewidth = 2)                                                      
                plt.title(gtitle)
                
            # Passes the if statement if the user entered a no(-adjecent) input
            elif stdcheck.lower() == "n" or stdcheck.lower() == "no":                                   
                pass
            else:
                
                # Runs the given print statement if anything other than yes or no is given
                print("Unknown input, assuming 'no'.")
                pass
        
        # Internal check function used to repeatedly ask for an input if an unaccepted one is given
        def histcheck():                                                                           
            
            # Creates an infinite loop until a numerical response is given                                                                                
            histcount = input("Do you want a histogram for 1 dataset, or 2 datasets? ")
            try:
                
                # Attempts to convert the histcheck input into an int variable
                histcount = int(histcount)
                
                # If conversion succeeds, checks if the value is not equal to 1 and/or 2
                if histcount != 1 and histcount != 2:
                    print("Please enter only 1 or 2.\n")
                    
                    # Returns the function to create a recursive function that will repeatedly ask for an input
                    # until an accepted one is entered
                    return histcheck()
                else:
                    return histcount
            except ValueError:
                print("Non-numerical entered. Please enter only 1 or 2.\n")
                
                # Returns the function to create a recursive function that will repeatedly ask for an input
                # until an accepted one is entered
                return histcheck()
        
        # Calls for and runs the histcheck function
        check = histcheck()
        
        ### For this entire method I'm not quite sure how to get both histograms to show the ###
        ### gaussian line when the user chooses to show both histograms at once. ###
        
        # Creates a continuous loop that breaks only if an accepted input is given
        while True:                                                                                                         
            
            # Checks if the histcheck value is 1
            if check == 1:                                                                                             
                while True:
                    
                    # If the above is true, calls for user input with the given printed statement
                    histnum = input(f"Which dataset would you like to use? {user_data._colname1} or {user_data._colname2}: ")   
                    
                    # Checks if the user input is equivalent to colname1
                    if histnum == user_data._colname1:                                                                        
                        
                        # If the above is true, creates a histogram from the first column of the user_data DataFrame
                        datafile.hist(bins = len(datafile.axes[0]), grid = False, rwidth = .9,                              
                                                column = columns[0], color = 'green', density = True)
                        
                        # Runs stdcheckfunc to create a standard distribution graph (if user entered yes earlier)
                        stdcheckfunc()
                        
                        # Breaks out of the inner while loop                                                                                 
                        break
                    
                    # Checks if the user input is equivalent to colname2
                    elif histnum == user_data._colname2:                                                                      
                        
                        # If the above is true,  creates histogram from the second column of the user_data DataFrame
                        datafile.hist(bins = len(datafile.axes[0]), grid = False, rwidth = .9,                              
                                                column = columns[1], color = 'green', density = True)
                        
                        # Runs stdcheckfunc to create a standard distribution graph (if user entered yes earlier)
                        stdcheckfunc()    
                        
                        # Breaks out of the inner while loop                                                                                  
                        break
                    else:
                        
                        # If any other input is entered, prints the given statement before going back to the start of
                        # the while loop
                        print(f"Please enter only {user_data._colname1} or {user_data._colname2}")   
                
                # Breaks out of the outer while loop                           
                break
            
            # Checks if the histcheck instance is equal to 2
            elif check == 2:
                
                # Creates a subplot which is 1 graph wide and 2 graphs tall
                fig, axes = plt.subplots(nrows = 2, ncols = 1)                        
                
                # Attaches the data from the first column of the df to the top plot                                      
                datafile.hist(bins = len(datafile.axes[0]), grid = False, rwidth = .9,
                                        column = columns[0], color = 'green', ax = axes[0],
                                        density = True)
                
                # Runs stdcheckfunc to create a standard distribution graph (if user entered yes earlier)
                stdcheckfunc()
                
                # Attaches the data from the second column of the df to the bottom plot
                datafile.hist(bins = len(datafile.axes[1]), grid = False, rwidth = .9,                                      
                                        column = columns[1], color = 'c', ax = axes[1],
                                        density = True)
                
                # Runs stdcheckfunc to create a standard distribution graph (if user entered yes earlier)
                stdcheckfunc()
                
                # Breaks out of the outer while loop
                break
            else:
                # If any other value is given for the histcheck instance, prints the given statement
                print("Invalid value detected. Terminating program.")
                exit()

        plt.show()
    
    def sctrplot(user_data: Data, gtitle = "Graph", marktype = "D", markc = 'c', markedge = 'k'):
        """ Uses the given x_data and y_data to create a scatter plot
            via matplot.pyplot's scatter method. Customization options
            are available, similar to the original pyplot method.

            Parameters:
                user_data : Data
                    Requires the user to pass in an instance of
                    Data to make use of the user's data.
                    
                gtitle : str [optional]
                    The desired graph title. Defaults to "Graph".
                    
                marktype : str [optional]
                    The desired marker style. Defaults to "D".
                    See link for all available matplotlib markers:
                    https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
                    
                markc : str [optional]
                    The desired marker color. Defaults to 'c'.
                    See link for all matplotlob colors:
                    https://matplotlib.org/stable/gallery/color/named_colors.html
                    
                markedge : str [optional]
                    The desired marker edge color. Defaults to 'k'.
                    See link for all matplotlob colors:
                    https://matplotlib.org/stable/gallery/color/named_colors.html
        """
        
        # Local instances of user_data._x_data and user_data._y_data for ease of use
        x_data = user_data._x_data
        y_data = user_data._y_data
        
        # Sets a new pyplot figure with the 'constrained' layout
        plt.figure(num = 1, layout = 'constrained')
        
        # Generates a scatter plot using the given x_data for x and y_data for y
        # Optional customization options can be used to change the output graph
        plt.scatter(x = x_data, y = y_data, marker = marktype, c = markc, edgecolors = markedge)
        
        # Pulls user_data colname1 and colname2 for the x-axis label and
        # y-axis label respectively.
        plt.xlabel(user_data._colname1)
        plt.ylabel(user_data._colname2)
        
        # Uses the passed in gtitle argument to set the plot's title
        plt.title(gtitle)
        
        print("\nDisplaying graph using user's data...")
        
        # Displays the generated plot
        plt.show()
    
    def resid(user_data: Data, gtitle = "Graph"):
        """ Uses user_data._df to create a residuals scatter plot
            via the seaborn sns.residplot method. The graph's
            title can optionally be customized.

            Parameters:
                user_data : Data
                    Requires the user to pass in an instance of
                    Data to make use of the user's data.
                    
                gtitle : str [optional]
                    The desired graph title. Defaults to "Graph".
        """
        
        # Sets a new pyplot figure
        plt.figure(num = 1)
        
        # Generates a residual scatter plot with the DataFrame created from
        # the user's data. Sets x and y labels to user_data._colname1
        # and user_data._colname2 respectively
        sns.residplot(data = user_data._df, x = user_data._colname1, y = user_data._colname2)
        
        # Sets the plot's title to the passed in gtitle argument
        plt.title(gtitle)
        
        # Displays the generated plot
        plt.show()

# External function that can be called by the user if they wish to. Is used only inside Data
def csvreader()-> np.ndarray:
    print("Please choose a csv file:")
    tk = Tk()
    tk.withdraw()
    
    # Opens a File Explorer window where the user can visually select a csv file
    path = askopenfilename()
    
    # Prints the file path
    print(path)
    
    # Converts the csv file into a pandas DataFrame
    with open(path, "r") as f:
        
        # Assumes no header or index has been set in the csv file
        datafile = pd.read_csv(f, header = None, index_col = None)
        
        # Saves column count for later use
        colcount = len(datafile.axes[1])
        
        # Converts the df into a numpy array
        dataarray = np.array(datafile)

    # Checks if the dimension of the array is equal to 1
    if np.ndim(dataarray) == 1:
        
        # Assumes given data is for the x_data and passes it into the x_data variable
        print("csv read into x_data")
        x_data = dataarray
        print("Note: y_data set to np.zeros_like(x_data)")
        y_data = np.zeros_like(x_data)
    
    # Checks if the array's dimensions are greater than 1
    elif np.ndim(dataarray) > 1:
        
        # Repeats the above steps if colcount is equal to 1
        if colcount == 1:
            print("csv read into x_data")
            x_data = dataarray[:,0]
        
        # Passes the first two columns into x_data and y_data if colcount is greater than 1
        elif colcount > 1:
            x_data = dataarray[:,0]
            y_data = dataarray[:,1]
    
    # Returns x_data and y_data
    return x_data, y_data
