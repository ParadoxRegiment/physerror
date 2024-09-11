import numpy as np
from numpy.typing import ArrayLike
from matplotlib import pyplot as plt
import scipy.stats as stats
from scipy.integrate import solve_ivp 
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pandas as pd
from dataclasses import dataclass, field
import seaborn as sns
import sys
import inquirer

@dataclass
class Data():
    """ An initializer and container dataclass that is initialized and reused
    by the user and other classes, as well as their methods. There are many 
    attributes that calculate common statistical errors, constants, and 
    general error propagation methods, and two class methods.
        
    Attributes
    ----------
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
    
    Parameters
    ----------
    user_x_data : ArrayLike = [1,2,3,4,5]
        []
    user_y_data : ArrayLike = [1,2,3,4,5]
        []
    data_type : str = 'manual'
        []
    
    Note
    ----
    Will update this soon-ish
    """
    user_x_data : ArrayLike = field(default_factory = lambda : [1,2,3,4,5])
    user_y_data : ArrayLike = field(default_factory = lambda : [1,2,3,4,5])
    data_type : str = field(default_factory = lambda : 'manual')
    _x_data : np.ndarray = field(init=False)
    _y_data : np.ndarray = field(init=False)
    _df : pd.DataFrame = field(init=False)
    _colname1 : str = field(init=False)
    _colname2 : str = field(init=False)
    
    def __post_init__(cls):
        cls._x_data, cls._y_data, cls._df, \
            cls._colname1, cls._colname2 = cls._initdata(cls.user_x_data, cls.user_y_data)
        cls.user_x_data = cls._x_data
        cls.user_y_data = cls._y_data
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
            cls.sigma_x_mean = cls.sigma_x/np.sqrt(N)
            cls.sigma_y_mean = cls.sigma_y/np.sqrt(N)
            cls.sigma_frac = 1/np.sqrt(2 * (N - 1))
    
    # Initializes and returns the data that will be reused
    def _initdata(cls, xdata = np.arange(5) + 1, ydata = np.arange(5) + 1):
        """ Callable but largely useless if done so. Used to read in a csv if
        the user so wishes, store data and their user-inputed names, then
        returns that data back to the line where it was called.
            
        Parameters
        ----------
        xdata : np.ndarray
            The given x data. As it is used, it is the "user_x_data"
            passed in at the time of initialization.
        
        ydata : np.ndarray
            The given y data. As it is used, it is the "user_y_data"
            pass in at the time of initialization.
    
        Returns
        -------
        np.ndarray
            The x_data array created from either the passed-in
            user_x_data or the csv file that was read in by the user.
        
        np.ndarray
            The y_data array created from either the passed-in
            user_y_data or the csv file that was read in by the user.
        
        pd.DataFrame
            The pandas DataFrame created from either the passed-in
            user_x_data and user_y_data or the csv file that was
            read in by the user.
        
        str
            The name entered by the user for x_data.
        
        str
            The name entered by the user for y_data.
        """
        reader = FileReaders()
        match cls.data_type:
            case 'CSV':
                x_data, y_data = reader.csvreader()
            case 'Excel':
                x_data, y_data = reader.excelreader()
            case 'manual':
                if type(xdata) == np.ndarray:
                    x_data = xdata
                    y_data = ydata
                else:
                    x_data = np.array(xdata)
                    y_data = np.array(xdata)
            case _:
                sys.exit("Unknown data type or WIP Section, please restart.")
        
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

        Returns
        -------
        np.ndarray
            An array that contains either the outliers that were found
            in the user's x_data, or a string stating no outliers were
            found.
        
        np.ndarray
            An array that contains either the outliers that were found
            in the user's y_data, or a string stating no outliers were
            found.
        """
        
        ### For some reason this is throwing an IndexError when deleting 
        
        # New x_data and y_data variables for ease of use
        x_data = cls._x_data
        y_data = cls._y_data
        print(x_data)
        print(y_data)
        
        # Immediately exits the program if the x_data or y_data are somehow empty arrays
        if np.size(x_data) == 0 or np.size(y_data) == 0:
            sys.exit()
        
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
                i -= 1
                
            # Checks if the row value is less than the mean - 2*sigma
            elif row < (cls.x_mean - 2*cls.sigma_x):    
                
                # If above is true, inserts the row value into the j cell of x_outliers
                x_outliers[j] = int(row)                    
                
                # Iterates j by one
                j += 1                                      
                
                # Deletes the outlier cell from x_data
                x_data = np.delete(x_data, i)
                i -= 1
                
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
                k -= 1
            
            # Checks if the row value is less than the mean - 2*sigma
            elif row < (cls.y_mean - 2*cls.sigma_y):    
                
                # If above is true, inserts the row value into the l cell of y_outliers
                y_outliers[l] = int(row)                    
                
                # Iterates j by one
                l += 1                                      
                
                # Deletes the outlier cell from y_data
                y_data = np.delete(y_data, k)
                k -= 1
                
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
    
    def export(cls):
        """Exports error analysis calculations to either an Excel workbook
        or JSON file based on user's choice.
        """
        import inquirer
        cls_dict = vars(cls)
        cls_dict_keys = cls_dict.keys()
        cls_keys_list = []
        cls_values_spliced_list = []
        for var in cls_dict_keys:
            if any(var_type in var for var_type in ['data', 'df', 'col']):
                pass
            else:
                cls_keys_list.append(var)
                cls_values_spliced_list.append(cls_dict[var])
        export_df = pd.DataFrame(np.transpose([cls_values_spliced_list]), index=cls_keys_list, columns=["Error Calculations"])
        print(export_df, "\n")
        
        file_type_q = [
            inquirer.List(
                "export_file",
                message="Choose a file type to export to",
                choices=["Excel",
                         "JSON"],
                ),
        ]
        
        match inquirer.prompt(file_type_q)["export_file"]:
            case "Excel":
                file_name = input("Enter a file name (no extension): ")
                file_path = file_name + ".xlsx"
                export_df.to_excel(file_path)
            case "JSON":
                file_name = input("Enter a file name (no extension): ")
                file_path = file_name + ".json"
                export_df.to_json(file_path, orient='columns', indent=4)
        
        print("File exported successfully.\n")

class Graphs:
    """Allows the user to create various graphs from the user_data
    pulled from Data.
        
    Attributes
    ----------
    graph_title : str = "Graph"
        String used as the title for any graphing function that is run.
        Defaults to "Graph".
    
    title_size : int = 11 | float
        Numerical value used as the title font size for any graphing
        function that is run. Defaults to 11.
    
    x_label : str = "x label"
        String used as the x-axis label for any graphing function with
        points that is run. Defaults to "x label"
    
    y_label : str = "y label"
        String used as the y-axis label for any graphing function with
        points that is run. Defaults to "y label".
        
    p_color : str = "cyan" | list[str] | ArrayLike[str]
        Either a single string, list, or ArrayLike of strings that is used
        to color the points in any graphing function with points that is run.
        Defaults to "cyan".
    
    line_color : str = "black"
        A string used to color the line in the linreg and resid functions.
        Defaults to "black".
    
    errbar_color : str = "red"
        A string used to color the error bars in the errbargraph function.
        Defaults to "red".
    
    dist_check : str = "Yes" | "No"
        A string used to determine if the datahist function will generate
        normal distrbution histograms. Only accepts "Yes" and "No", and
        defaults to "No".
    
    dataset_check : int = 1 | 2
        An integer used to determine how many datasets will be generated
        from the datahist function. Only accepts 1 and 2, and defaults to 1.
    
    hist_color : str = "green" | list[str] | ArrayLike[str]
        Either a single string, list, or ArrayLike of strings to determine
        the color of the the histogram(s) from the datahist function. Defaults
        to "green".
    
    dbl_pend_line : str = "lime"
        A string used to change the color of specifically the double pendulum
        function's line color. Defaults to "lime".
    
    dbl_pend_trace : str = "black"
        A string used to change the color of specifically the double pendulum
        function's trace line color. Defaults to "black".
    """
    def __init__(cls):
        cls.graph_title = "Graph"
        cls.title_size = 11
        cls.x_label = "x label"
        cls.y_label = "y label"
        cls.p_color = 'red'
        cls.line_color = 'black'
        cls.errbar_color = 'red'
        cls.dist_check = 'No'
        cls.dataset_check = 1
        cls.hist_color = 'green'
        cls.dbl_pend_line = 'lime'
        cls.dbl_pend_trace = 'black'
    
    def linreg(cls, user_data : Data):
        """ Uses the given x_data and y_data arrays to create a linear
        regression plot.
            
        Parameters
        ----------
        user_data : Data
            Requires the user to pass in an instance of
            Data to make use of the user's data.
        """
        
        # New x_data and y_data for ease of use
        x_data = user_data._x_data                                
        y_data = user_data._y_data
        
        # Sets the figure's title to the default (or passed in) graph title
        plt.title(cls.graph_title, fontsize = cls.title_size)
        
        # Sets the figure data to x_data and y_data, colored orange
        if np.size(cls.p_color) == 1: 
            plt.plot(x_data, y_data, 'o', color = cls.p_color, linestyle="")
        elif np.size(cls.p_color) != 1:
            for i in range(np.size(cls.p_color)):
                plt.plot(x_data[i], y_data[i], 'o', color = cls.p_color[i])
        
        # Sets the figure's xlabel to the user's entered x_data name
        plt.xlabel(cls.x_label, fontsize = 11)
        
        # Sets the figure's xlabel to the user's entered y_data name
        plt.ylabel(cls.y_label, fontsize = 11)
        
        # Adds the linear regression line to the plot
        plt.plot(x_data, user_data.A + user_data.B * x_data, color = cls.line_color)
        
        # Displays the linear regression plot
        plt.show()
    
    def errbargraph(cls, user_data : Data):
        """ Uses the x data and y data from Data to create a point-based
        with error bars on each point. Error size is the sigma_x value
        calculated in Data.
            
        Parameters
        ----------
        user_data : Data
            Requires the user to pass in an instance of
            Data to make use of the user's data.
        """
        
        # New df for ease of use
        # df = user_data._df
        x_data = user_data._x_data
        y_data = user_data._y_data
        
        plt.title(cls.graph_title, fontsize = cls.title_size)
        plt.xlabel(cls.x_label, fontsize=11)
        plt.ylabel(cls.y_label, fontsize=11)
            
        # Checks the size of the p_color attribute and plots the graph depending on if it's
        # greater than 1.
        if np.size(cls.p_color) == 1:
            plt.errorbar(x=x_data, y=y_data, yerr=user_data.sigma_x, marker=".",
                         capsize=3, ecolor=cls.errbar_color, linewidth=1, color=cls.p_color,
                         linestyle="")
        elif np.size(cls.p_color) > 1:
            for i in range(np.size(x_data)):
                plt.errorbar(x=x_data[i], y=y_data[i], yerr=user_data.sigma_x, marker=".",
                            capsize=3, ecolor=cls.errbar_color, linewidth=1, color=cls.p_color[i])
        plt.show()
    
    def datahist(cls, user_data : Data):
        """Generates a histogram of one or two sets of data pulled
        from the Data class using pandas' DataFrame.hist method.
        
        Parameters
        ----------
        user_data : Data
            Requires the user to pass in an instance of
            Data to make use of the user's data.
        """
        # New df for ease of use
        datafile = user_data._df                                                                          
        
        # Initialized to the df's columns array for future use
        columns = datafile.columns                                                                      
        
        stdcheck = cls.dist_check
        
        # Internal function used to determine what type of histogram graph will be created
        def stdcheckfunc(ax = None, index = 0):
            
            # Checks if the user entered a yes(-adjecent) input
            if stdcheck.lower() == "y" or stdcheck.lower() == "yes":
                
                # Pulls the minimum and maximum for the x-axis from data later down
                if index == 0:
                    xmin = np.min(user_data.user_x_data)
                    xmax = np.max(user_data.user_x_data)
                    mean = user_data.x_mean
                    sigma = user_data.sigma_x
                else:
                    xmin = np.min(user_data.user_y_data)
                    xmax = np.max(user_data.user_y_data)
                    mean = user_data.y_mean
                    sigma = user_data.sigma_y

                # Creates an x-axis array of 100 values
                x = np.linspace(xmin, xmax, 100)
                
                # Calls the scipy.stats.pdf method with x as the data, user_data x_mean as the mean,
                # and user_data sigma_x as the scale
                p = stats.norm.pdf(x, mean, sigma)
                
                # Creates a graph with x on the x-axis and p on the y-axis
                if check == 1:
                    plt.plot(x, p, 'k', linewidth = 2)
                else:
                    ax.plot(x, p, 'k', linewidth = 2)
                # plt.title(cls.graph_title)
                
            # Passes the if statement if the user entered a no(-adjecent) input
            elif stdcheck.lower() == "n" or stdcheck.lower() == "no":                                   
                pass
            else:
                # Runs the given print statement if anything other than yes or no is given
                print("Unknown input, assuming 'no'.")
                pass
        
        # Calls for and runs the histcheck function
        check = cls.dataset_check
        
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
                        datafile.hist(bins = len(datafile.index), grid = False, rwidth = .9,                              
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
                if np.size(cls.hist_color) == 1:
                    local_hist_color = ['green', 'red']
                else:
                    local_hist_color = cls.hist_color

                # Creates a subplot which is 1 graph wide and 2 graphs tall
                fig, axes = plt.subplots(nrows = 2, ncols = 1)
                fig.suptitle(cls.graph_title, ha='center', va='center')
                for i in range(check):
                    ax = axes[i]

                    # Attaches the data from the first column of the df to the top plot
                    datafile.hist(bins = len(datafile.axes[0]), grid = False, rwidth = .9,
                                column = columns[i], color = local_hist_color[i], ax = ax,
                                density = True)
                    if i == 0:
                        # Runs stdcheckfunc to create a standard distribution graph (if user entered yes earlier)
                        stdcheckfunc(ax=ax, index = i)
                    elif i == 1:
                        stdcheckfunc(ax=ax, index = i)
                        # Breaks out of the outer while loop
                        break
                    
                # Breaks out of the outer while loop
                break
            else:
                # If any other value is given for the histcheck instance, prints the given statement
                print("Invalid value detected. Terminating program.")
                exit()

        plt.show()
    
    def sctrplot(cls, user_data: Data):
        """ Uses the given x_data and y_data to create a scatter plot
        via matplot.pyplot's scatter method. Customization options
        are available, similar to the original pyplot method.

        Parameters
        ----------
        user_data : Data
            Requires the user to pass in an instance of
            Data to make use of the user's data.
        """
        
        # Local instances of user_data._x_data and user_data._y_data for ease of use
        x_data = user_data._x_data
        y_data = user_data._y_data
        
        # Sets a new pyplot figure with the 'constrained' layout
        plt.figure(num = 1, layout = 'constrained')
        
        # Generates a scatter plot using the given x_data for x and y_data for y
        # Optional customization options can be used to change the output graph
        if np.size(cls.p_color) == 1:
            plt.scatter(x = x_data, y = y_data, marker = "D", c = cls.p_color, edgecolors = 'k')
        elif np.size(cls.p_color) != 1:
            for i in range(np.size(cls.p_color)):
                plt.scatter(x = x_data[i], y = y_data[i], marker = "D", c = cls.p_color[i], edgecolors = 'k')
        
        # Pulls user_data colname1 and colname2 for the x-axis label and
        # y-axis label respectively.
        plt.xlabel(cls.x_label)
        plt.ylabel(cls.y_label)
        
        # Uses the passed in gtitle argument to set the plot's title
        plt.title(cls.graph_title)
        
        print("\nDisplaying graph using user's data...")
        
        # Displays the generated plot
        plt.show()
    
    def resid(cls, user_data: Data):
        """ Uses user_data._df to create a residuals scatter plot
        via the seaborn sns.residplot method. The graph's title
        can optionally be customized.

        Parameters
        ----------
        user_data : Data
            Requires the user to pass in an instance of
            Data to make use of the user's data.
        """
        
        # Sets a new pyplot figure
        plt.figure(num = 1)
        
        # Generates a residual scatter plot with the DataFrame created from
        # the user's data. Sets x and y labels to user_data._colname1
        # and user_data._colname2 respectively
        sns.residplot(data = user_data._df, x = cls.x_label, y = cls.y_label)
        
        # Sets the plot's title to the passed in gtitle argument
        plt.title(cls.graph_title)
        
        # Displays the generated plot
        plt.show()
        
    def dbl_pend(cls, theta_0 : float, phi_0 : float, theta_dot_0 = 0, phi_dot_0 = 0, anim_type = 0):
        """Generates either a point mass or bar mass double pendulum
        animation based on the pass in initial values. Angles are read
        as the angle between the bar/string and an imaginary horizontal
        line going through the point.
        
        Point mass calculations and animation code were taken from
        matplotlib's documentation:
        https://matplotlib.org/stable/gallery/animation/double_pendulum.html

        Parameters
        ----------
        theta_0 : float
            Initial angle of the top bar/string.
            
        phi_0 : float
            Initial angle of the bottom bar/string.
            
        theta_dot_0 : int = 0 (optional)
            Initial velocity of the top bar/string. Defaults to 0.
            
        phi_dot_0 : int = 0 (optional)
            Initial velocity of the bottom bar/string. Defaults to 0.
            
        anim_type : int = 0 | 1 (optional)
            Optional variable that determines the type of double
            pendulum that will be used. Defaults to 0 for Point Mass,
            accepts 1 for Bar Mass.
        """
        import matplotlib.animation as animation
        gravity = 9.81      # m/s^2
        
        len_1 = float(input('Enter the length of the top pendulum in meters: '))
        len_2 = float(input('Enter the length of the bottom pendulum in meters: '))
        tot_len = len_1 + len_2
        mass_1 = float(input('Enter the mass of the top pendulum in g: '))
        mass_2 = float(input('Enter the mass of the bottom pendulum in g: '))
        time_lim = float(input('Enter the time limit in seconds: '))
        
        def point_mass(time, state):
            dydx = np.zeros_like(state)
            
            dydx[0] = state[1]
            
            ang_delta = state[2] - state[0]
            den1 = (mass_1 + mass_2) * len_1 - mass_2 * len_1 * np.cos(ang_delta) * np.cos(ang_delta)
            
            dydx[1] = ((mass_2 * len_1 * state[1] * state[1] * np.sin(ang_delta) * np.cos(ang_delta) + mass_2 * gravity * np.sin(state[2]) * np.cos(ang_delta)
                        + mass_2 * len_2 * state[3] * state[3] * np.sin(ang_delta) - (mass_1 + mass_2) * gravity * np.sin(state[0])) / den1)
            
            dydx[2] = state[3]

            den2 = (len_2/len_1) * den1
            dydx[3] = ((- mass_2 * len_2 * state[3] * state[3] * np.sin(ang_delta) * np.cos(ang_delta) + (mass_1 + mass_2) * gravity * np.sin(state[0]) * np.cos(ang_delta)
                        - (mass_1 + mass_2) * len_1 * state[1] * state[1] * np.sin(ang_delta) - (mass_1 + mass_2) * gravity * np.sin(state[2])) / den2)
            
            return dydx

        def bar_mass(time, state):
            dydx = np.zeros_like(state)
            
            dydx[0] = state[1]
            
            ang_delta = state[2] - state[0]
            mu = 1 + (mass_1/mass_2)
            den1 = len_1*(mu - np.cos(ang_delta)**2)

            dydx[1] = (gravity*(np.sin(state[2])*np.cos(ang_delta) - mu*np.sin(state[0])) -
                        (len_2*(state[3]**2) + len_1*(state[1]**2)*np.cos(ang_delta))*np.sin(ang_delta))/den1
            
            dydx[2] = state[3]

            den2 = (len_2/len_1) * den1
            dydx[3] = (gravity*mu*(np.sin(state[0])*np.cos(ang_delta) - np.sin(state[2])) -
                        (mu*len_1*(state[1]**2) + len_2*(state[3]**2)*np.cos(ang_delta))*np.sin(ang_delta))/den2
            
            return dydx
        
        state = np.radians([theta_0, theta_dot_0, phi_0, phi_dot_0])
        dt = 0.03
        time = np.arange(0, time_lim, dt)
        
        if anim_type == 0:
            output = solve_ivp(point_mass, time[[0, -1]], state, t_eval=time).y.T
            anim_type = 'point'
        elif anim_type == 1:
            output = solve_ivp(bar_mass, time[[0, -1]], state, t_eval=time).y.T
            anim_type = 'bar'
        else:
            print('Invalid input, exiting program')
            exit()
        
        out_x1 = len_1 * np.sin(output[:, 0])
        out_y1 = -len_1 * np.cos(output[:, 0])
        out_x2 = len_2 * np.sin(output[:, 2]) + out_x1
        out_y2 = -len_2 * np.cos(output[:, 2]) + out_y1

        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(autoscale_on=False, xlim=(-tot_len, tot_len), ylim=(-tot_len, tot_len))
        ax.set_aspect('equal')
        ax.grid()

        line, = ax.plot([], [], 'o-', color=cls.dbl_pend_line, lw=2)
        trace, = ax.plot([], [], 'k.-', color=cls.dbl_pend_trace, lw=1, ms=2)
        time_template = 'time = %.2fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
        
        def animate(i):
            thisx = [0, out_x1[i], out_x2[i]]
            thisy = [0, out_y1[i], out_y2[i]]

            history_x = out_x2[:i]
            history_y = out_y2[:i]

            line.set_data(thisx, thisy)
            trace.set_data(history_x, history_y)
            time_text.set_text(time_template % (i*dt))
            return line, trace, time_text

        plt.title(cls.graph_title, fontsize=cls.title_size)
        ani = animation.FuncAnimation(fig, animate, len(output), interval=dt*1000, blit=True)
        print(f"Saving gif as {theta_0}{phi_0}{anim_type}anim.gif")
        ani.save(f"{str(theta_0).strip('.')}{str(phi_0).strip('.')}{anim_type}anim.gif", writer='imagemagick', fps=20)
        plt.show()

class _InquirePrompts:
    """A private class that contains all the methods and functions needed
    to create and run the command-line-interface.
    """
    def __init__(cls):
        cls.graphs_obj = Graphs()
        
        cls._title_prompt = "Title"
        cls._title_size_prompt = "Title Size"
        cls._x_label_prompt = "x-label"
        cls._y_label_prompt = "y-label"
        cls._point_colors_prompt = "Point Colors"
        cls._line_color_prompt = "Line Color"
        cls._errbar_color_prompt = "Error Bar Color"
        cls._hist_color_prompt = "Histogram Color(s)"
        cls._theta_0 = 122
        cls._phi_0 = 122
        cls._theta_0_dot = 0
        cls._phi_0_dot = 0
        cls._pend_type = 0
        
        cls.data_q = [
                inquirer.List(
                    "data",
                    message="Select a data file type",
                    choices=["CSV", "Excel"],
                    ),
        ]
        
        cls.funcs_q = [
            inquirer.List(
                "function",
                message="Select a function to modify and/or run",
                choices=["Export Data",
                         "Data.outlier",
                         "Graphs.linreg",
                         "Graphs.errbargraph",
                         "Graphs.datahist",
                         "Graphs.sctrplot",
                         "Graphs.resid",
                         "Graphs.dbl_pend",
                         "Change File/Data",
                         "Exit/Quit"],
                ),
        ]
        
        ufunc_msg = "Select a property to change, or 'Run' to run the function. Select 'Back' to return to the function select menu."
        cls.linreg_q = [
            inquirer.List(
                "linreg",
                message="Graphs.linreg -- " + ufunc_msg,
                choices=[cls._title_prompt,
                         cls._title_size_prompt,
                         cls._x_label_prompt,
                         cls._y_label_prompt,
                         cls._point_colors_prompt,
                         cls._line_color_prompt,
                         "Run", "Back", "Exit/Quit"],
                ),
        ]
        
        cls.errbar_q = [
            inquirer.List(
                "errbar",
                message="Graphs.errbargraph -- " + ufunc_msg,
                choices=[cls._title_prompt,
                         cls._title_size_prompt,
                         cls._x_label_prompt,
                         cls._y_label_prompt,
                         cls._point_colors_prompt,
                         cls._errbar_color_prompt,
                         "Run", "Back", "Exit/Quit"],
                ),
        ]
        
        cls.datahist_q = [
            inquirer.List(
                "datahist",
                message="Graphs.datahist -- " + ufunc_msg,
                choices=[cls._title_prompt,
                         cls._title_size_prompt,
                         "Normal Distribution",
                         cls._hist_color_prompt,
                         "Dataset Count",
                         "Run", "Back", "Exit/Quit"],
                ),
        ]
        
        cls.datahist_count = [
            inquirer.List(
                "datahist_count",
                message="Select a number of data sets to use",
                choices=['1',
                         '2']
            ),
        ]
        
        cls.datahist_normal = [
            inquirer.List(
                "datahist_normal",
                message="Select Yes, No, or Cancel",
                choices=['Yes',
                         'No',
                         'Cancel']
            )
        ]
        
        cls.sctrplot_q = [
            inquirer.List(
                "sctrplot",
                message="Graphs.sctrplot -- " + ufunc_msg,
                choices=[cls._title_prompt,
                         cls._title_size_prompt,
                         cls._x_label_prompt,
                         cls._y_label_prompt,
                         cls._point_colors_prompt,
                         "Run", "Back", "Exit/Quit"],
                ),
        ]
        
        cls.resid_q = [
            inquirer.List(
                "resid",
                message="Graphs.resid -- " + ufunc_msg,
                choices=[cls._title_prompt,
                         cls._title_size_prompt,
                         cls._x_label_prompt,
                         cls._y_label_prompt,
                         cls._point_colors_prompt,
                         "Run", "Back", "Exit/Quit"],
                ),
        ]
        
        cls.dbl_pend_q = [
            inquirer.List(
                "dblpend",
                message="Graphs.dbl_pend -- " + ufunc_msg,
                choices=[cls._title_prompt,
                         cls._title_size_prompt,
                         "Line Color",
                         "Trace Color",
                         "Initial Values",
                         "Pendulum Type",
                         "Run", "Back", "Exit/Quit"],
                ),
        ]
        
        cls.dbl_pend_init_q = [
            inquirer.List(
                "dblpend_init",
                message="Choose an initial value to change",
                choices=["Theta 0",
                         "Theta 0 Dot",
                         "Phi 0",
                         "Phi 0 Dot",
                         "Back", "Exit/Quit"],
                ),
        ]
        
        cls.dbl_pend_type_q = [
            inquirer.List(
                "dblpend_type",
                message="Choose an pendulum type",
                choices=["Point Mass",
                         "Bar Mass"],
                ),
        ]
        
        cls.val_change_q = [
            inquirer.List(
                "val_change",
                message="Would you like to edit this property?",
                choices=["Yes",
                         "No",
                        "Exit/Quit"],
                ),
        ]
        
        cls.p_color_q = [
            inquirer.List(
                "p_color",
                message="Single color or multiple colors?",
                choices=["Single",
                         "Multiple",
                         "Back", "Exit/Quit"],
                ),
        ]
        
        cls.cont_check = [
            inquirer.List(
                "cont_check",
                message="Would you like to run another function?",
                choices=["Yes",
                         "No"],
                ),
        ]
        
    def inq_prompt(cls, cls_obj : object):
        """Large set of inquiry prompt functions that interconnect to create
        a smooth, cohesive command-line-interface for ease of use, particularly
        for users with little to no coding experience.
        """
        # data_ans will later be used to check what type of data file will be
        # read into the Data class
        _stored_values = cls_obj
        data_ans = inquirer.prompt(cls.data_q)['data']
        tempdata = Data(data_type=data_ans)
        
        def cont_check_prompt():
            """Used to verify whether or not the user wants to keep the program
            running to use other functions.
            """
            match inquirer.prompt(cls.cont_check)["cont_check"]:
                case "Yes":
                    func_prompts()
                case "No":
                    sys.exit("Closing program...")
        
        def val_change_prompt(val_type : str, curr_val):
            """Function used to check against what type of variable is being
            changed so the proper variable type can be returned. i.e. making
            sure that the title size is returned as a numerical rather than
            a string.
            """
            match inquirer.prompt(cls.val_change_q)["val_change"]:
                case "Yes":
                    match val_type:
                        case "title" | "label":  
                            return input("New title value: ")
                        case "size":
                            new_val = input("New size value: ")
                            try:
                                float(new_val)
                                return new_val
                            except ValueError:
                                print("\nOnly numerical values are accepted. Resetting to previous value.\n")
                                return curr_val
                        case "line":
                            return input("New color value: ")
                        case "point":
                            match inquirer.prompt(cls.p_color_q)["p_color"]:
                                case "Single":
                                    return input("New color value: ")
                                case "Multiple":
                                    p_color_list = input(f"Choose {np.size(tempdata._x_data)} colors, separated by a comma\
                                                         and space. Repeat colors are allowed:\n").split(', ')
                                    if np.size(p_color_list) > np.size(tempdata._x_data):
                                        print('\nToo many values entered. Resetting to last value.\n')
                                        return curr_val
                                    else:
                                        return p_color_list
                        case 'bar':
                            match cls.graphs_obj.dataset_check:
                                case 1:
                                    bar_color = input(f"Choose a color for the histogram\n")
                                    return bar_color
                                case 2:
                                    bar_color = input(f"Choose 2 colors for the histograms, separated by a comma and a space.\n").split(', ')
                                    if np.size(bar_color) > 2:
                                        print("\nToo many colors entered, resetting to the original value.")
                                        return curr_val
                                    else:
                                        return bar_color
                        case "angle" | "velocity":
                            new_val = input(f"New {val_type} value. If this is an angle, use in degrees: ")
                            try:
                                float(new_val)
                                return new_val
                            except ValueError:
                                print("\nOnly numerical values are accepted. Resetting to previous value.\n")
                                return curr_val
                case "No":
                    return curr_val
                case "Exit/Quit":
                    sys.exit("Closing program...")

        def linreg_prompts():
            match inquirer.prompt(cls.linreg_q)["linreg"]:
                case cls._title_prompt:
                    curr_title = cls.graphs_obj.graph_title
                    print(f"Current title: {curr_title}")
                    cls.graphs_obj.graph_title = val_change_prompt('title', curr_title)
                    return linreg_prompts()
                case cls._title_size_prompt:
                    curr_size = cls.graphs_obj.title_size
                    print(f"Current size: {curr_size}")
                    cls.graphs_obj.title_size = val_change_prompt('size', curr_size)
                    return linreg_prompts()
                case cls._x_label_prompt:
                    curr_label = cls.graphs_obj.x_label
                    print(f"Current label: {curr_label}")
                    cls.graphs_obj.x_label = val_change_prompt('label', curr_label)
                    return linreg_prompts()
                case cls._y_label_prompt:
                    curr_label = cls.graphs_obj.y_label
                    print(f"Current label: {curr_label}")
                    cls.graphs_obj.y_label = val_change_prompt('label', curr_label)
                    return linreg_prompts()
                case cls._point_colors_prompt:
                    curr_p_color = cls.graphs_obj.p_color
                    print(f"Current color(s): {curr_p_color}")
                    cls.graphs_obj.p_color = val_change_prompt('point', curr_p_color)
                    return linreg_prompts()
                case cls._line_color_prompt:
                    curr_line_color = cls.graphs_obj.line_color
                    print(f"Current line color: {curr_line_color}")
                    cls.graphs_obj.line_color = val_change_prompt('line', curr_line_color)
                    return linreg_prompts()
                case "Run":
                    cls.graphs_obj.linreg(tempdata)
                    cont_check_prompt()
                case "Back":
                    func_prompts()
                case "Exit/Quit":
                    sys.exit("Closing program...")
        
        def errbar_prompts():
            match inquirer.prompt(cls.errbar_q)["errbar"]:
                case cls._title_prompt:
                    curr_title = cls.graphs_obj.graph_title
                    print(f"Current title: {curr_title}")
                    cls.graphs_obj.graph_title = val_change_prompt('title', curr_title)
                    return errbar_prompts()
                case cls._title_size_prompt:
                    curr_size = cls.graphs_obj.title_size
                    print(f"Current size: {curr_size}")
                    cls.graphs_obj.title_size = val_change_prompt('size', curr_size)
                    return errbar_prompts()
                case cls._x_label_prompt:
                    curr_label = cls.graphs_obj.x_label
                    print(f"Current label: {curr_label}")
                    cls.graphs_obj.x_label = val_change_prompt('label', curr_label)
                    return errbar_prompts()
                case cls._y_label_prompt:
                    curr_label = cls.graphs_obj.y_label
                    print(f"Current label: {curr_label}")
                    cls.graphs_obj.y_label = val_change_prompt('label', curr_label)
                    return errbar_prompts()
                case cls._point_colors_prompt:
                    curr_p_color = cls.graphs_obj.p_color
                    print(f"Current color(s): {curr_p_color}")
                    cls.graphs_obj.p_color = val_change_prompt('point', curr_p_color)
                    return errbar_prompts()
                case cls._errbar_color_prompt:
                    curr_errbar_color = cls.graphs_obj.errbar_color
                    print(f"Current line color: {curr_errbar_color}")
                    cls.graphs_obj.errbar_color = val_change_prompt('line', curr_errbar_color)
                    return errbar_prompts()
                case "Run":
                    cls.graphs_obj.errbargraph(tempdata)
                    cont_check_prompt()
                case "Back":
                    func_prompts()
                case "Exit/Quit":
                    sys.exit("Closing program...")
        
        # "Title", "Title Size", "Normal Distribution", "Bar Color", "Dataset Count", "Run", "Back"
        def datahist_prompts():
            match inquirer.prompt(cls.datahist_q)["datahist"]:
                case cls._title_prompt:
                    curr_title = cls.graphs_obj.graph_title
                    print(f"Current title: {curr_title}")
                    cls.graphs_obj.graph_title = val_change_prompt('title', curr_title)
                    return datahist_prompts()
                case cls._title_size_prompt:
                    curr_size = cls.graphs_obj.title_size
                    print(f"Current size: {curr_size}")
                    cls.graphs_obj.title_size = val_change_prompt('size', curr_size)
                    return datahist_prompts()
                case "Normal Distribution":
                    curr_norm = cls.graphs_obj.dist_check
                    print(f"Current selection: {curr_norm}")
                    temp_norm = inquirer.prompt(cls.datahist_normal)['datahist_normal']
                    if temp_norm == 'Cancel':
                        return datahist_prompts()
                    else:
                        cls.graphs_obj.dist_check = temp_norm
                        return datahist_prompts()
                case cls._hist_color_prompt:
                    curr_hist_color = cls.graphs_obj.hist_color
                    print(f"Current color(s): {curr_hist_color}")
                    cls.graphs_obj.hist_color = val_change_prompt('bar', curr_hist_color)
                    return datahist_prompts()
                case "Dataset Count":
                    curr_count = cls.graphs_obj.dataset_check
                    print(f"Current count: {curr_count}")
                    temp_count = inquirer.prompt(cls.datahist_count)['datahist_count']
                    cls.graphs_obj.dataset_check = int(temp_count)
                    return datahist_prompts()
                case "Run":
                    cls.graphs_obj.datahist(tempdata)
                    cont_check_prompt()
                case "Back":
                    func_prompts()
                case "Exit/Quit":
                    sys.exit("Closing program...")
        
        # "Title", "Title Size", "x-label", "y-label", "Point Color(s)", "Run", "Back"
        def sctrplot_prompts():
            match inquirer.prompt(cls.linreg_q)["linreg"]:
                case cls._title_prompt:
                    curr_title = cls.graphs_obj.graph_title
                    print(f"Current title: {curr_title}")
                    cls.graphs_obj.graph_title = val_change_prompt('title', curr_title)
                    return sctrplot_prompts()
                case cls._title_size_prompt:
                    curr_size = cls.graphs_obj.title_size
                    print(f"Current size: {curr_size}")
                    cls.graphs_obj.title_size = val_change_prompt('size', curr_size)
                    return sctrplot_prompts()
                case cls._x_label_prompt:
                    curr_label = cls.graphs_obj.x_label
                    print(f"Current label: {curr_label}")
                    cls.graphs_obj.x_label = val_change_prompt('label', curr_label)
                    return sctrplot_prompts()
                case cls._y_label_prompt:
                    curr_label = cls.graphs_obj.y_label
                    print(f"Current label: {curr_label}")
                    cls.graphs_obj.y_label = val_change_prompt('label', curr_label)
                    return sctrplot_prompts()
                case cls._point_colors_prompt:
                    curr_p_color = cls.graphs_obj.p_color
                    print(f"Current color(s): {curr_p_color}")
                    cls.graphs_obj.p_color = val_change_prompt('point', curr_p_color)
                    return sctrplot_prompts()
                case "Run":
                    cls.graphs_obj.sctrplot(tempdata)
                    cont_check_prompt()
                case "Back":
                    func_prompts()
                case "Exit/Quit":
                    sys.exit("Closing program...")
        
        # "Title", "Title Size", "x-label", "y-label", "Point Color(s)", "Line Color", "Run", "Back"
        def resid_prompts():
            match inquirer.prompt(cls.linreg_q)["linreg"]:
                case cls._title_prompt:
                    curr_title = cls.graphs_obj.graph_title
                    print(f"Current title: {curr_title}")
                    cls.graphs_obj.graph_title = val_change_prompt('title', curr_title)
                    return linreg_prompts()
                case cls._title_size_prompt:
                    curr_size = cls.graphs_obj.title_size
                    print(f"Current size: {curr_size}")
                    cls.graphs_obj.title_size = val_change_prompt('size', curr_size)
                    return linreg_prompts()
                case cls._x_label_prompt:
                    curr_label = cls.graphs_obj.x_label
                    print(f"Current label: {curr_label}")
                    cls.graphs_obj.x_label = val_change_prompt('label', curr_label)
                    return linreg_prompts()
                case cls._y_label_prompt:
                    curr_label = cls.graphs_obj.y_label
                    print(f"Current label: {curr_label}")
                    cls.graphs_obj.y_label = val_change_prompt('label', curr_label)
                    return linreg_prompts()
                case cls._point_colors_prompt:
                    curr_p_color = cls.graphs_obj.p_color
                    print(f"Current color(s): {curr_p_color}")
                    cls.graphs_obj.p_color = val_change_prompt('point', curr_p_color)
                    return linreg_prompts()
                case cls._line_color_prompt:
                    curr_line_color = cls.graphs_obj.line_color
                    print(f"Current line color: {curr_line_color}")
                    cls.graphs_obj.line_color = val_change_prompt('line', curr_line_color)
                    return linreg_prompts()
                case "Run":
                    cls.graphs_obj.linreg(tempdata)
                    cont_check_prompt()
                case "Back":
                    func_prompts()
                case "Exit/Quit":
                    sys.exit("Closing program...")
        
        def dbl_pend_prompts():
            print("Note: This menu will allow you to change the initial angles and velocities",
                  "\nand graph properties. When you run the function it will ask you to manually",
                  "\nenter multiple other initial values, such as length and weight.\n")
            
            match inquirer.prompt(cls.dbl_pend_q)["dblpend"]:
                case cls._title_prompt:
                    curr_title = cls.graphs_obj.graph_title
                    print(f"Current title: {curr_title}")
                    cls.graphs_obj.graph_title = val_change_prompt('title', curr_title)
                    return dbl_pend_prompts()
                case cls._title_size_prompt:
                    curr_size = cls.graphs_obj.title_size
                    print(f"Current size: {curr_size}")
                    cls.graphs_obj.title_size = val_change_prompt('size', curr_size)
                    return dbl_pend_prompts()
                case "Line Color":
                    curr_pend_line_color = cls.graphs_obj.dbl_pend_line
                    print(f"Current line color: {curr_pend_line_color}")
                    cls.graphs_obj.dbl_pend_line = val_change_prompt('line', curr_pend_line_color)
                    return dbl_pend_prompts()
                case "Trace Color":
                    curr_trace_color = cls.graphs_obj.dbl_pend_trace
                    print(f"Current trace: {curr_trace_color}")
                    cls.graphs_obj.dbl_pend_trace = val_change_prompt('line', curr_trace_color)
                    return dbl_pend_prompts()
                case "Initial Values":
                    dbl_pend_init_prompts()
                case "Pendulum Type":
                    if cls._pend_type == 0:
                        curr_pend_type = "Point Mass"
                    elif cls._pend_type == 1:
                        curr_pend_type = "Bar Mass"
                    print(f"Current pendulum type: {curr_pend_type}")
                    match inquirer.prompt(cls.dbl_pend_type_q)["dblpend_type"]:
                        case "Point Mass":
                            cls._pend_type = 0
                            return dbl_pend_prompts()
                        case "Bar Mass":
                            cls._pend_type = 1
                            return dbl_pend_prompts()
                case "Run":
                    cls.graphs_obj.dbl_pend(cls._theta_0, cls._phi_0, cls._theta_0_dot, cls._phi_0_dot)
                    cont_check_prompt()
                case "Back":
                    func_prompts()
                case "Exit/Quit":
                    sys.exit("Closing program...")
        
        def dbl_pend_init_prompts():
            match inquirer.prompt(cls.dbl_pend_init_q)["dblpend_init"]:
                case "Theta 0":
                    curr_theta = cls._theta_0
                    print(f"Current theta 0: {curr_theta}")
                    cls._theta_0 = val_change_prompt('angle', curr_theta)
                    return dbl_pend_init_prompts()
                case "Theta 0 Dot":
                    curr_theta_dot = cls._theta_0_dot
                    print(f"Current theta 0 dot: {curr_theta_dot}")
                    cls._theta_0_dot = val_change_prompt('velocity', curr_theta_dot)
                    return dbl_pend_init_prompts()
                case "Phi 0":
                    curr_phi = cls._phi_0
                    print(f"Current phi 0: {curr_phi}")
                    cls._phi_0 = val_change_prompt('angle', curr_phi)
                    return dbl_pend_init_prompts()
                case "Phi 0 Dot":
                    curr_phi_dot = cls._phi_0_dot
                    print(f"Current phi 0 dot: {curr_phi_dot}")
                    cls._phi_0_dot = val_change_prompt('velocity', curr_phi_dot)
                    return dbl_pend_init_prompts()
                case "Back":
                    return dbl_pend_prompts()
                case "Exit/Quit":
                    sys.exit("Closing program...")
        
        def func_prompts():
            print(_stored_values._theta_0)
            funcs_ans = inquirer.prompt(cls.funcs_q)
            match funcs_ans["function"]:
                case "Export Data":
                    tempdata.export()
                    cont_check_prompt()
                case "Data.outlier":
                    xout, yout = tempdata.outlier()
                    print(f"x outliers: {xout}")
                    print(f"y outliers: {yout}")
                    cont_check_prompt()
                case "Graphs.linreg":
                    linreg_prompts()
                case "Graphs.errbargraph":
                    errbar_prompts()
                case "Graphs.datahist":
                    datahist_prompts()
                case "Graphs.sctrplot":
                    sctrplot_prompts()
                case "Graphs.resid":
                    resid_prompts()
                case "Graphs.dbl_pend":
                    dbl_pend_prompts()
                case "Change File/Data":
                    restart = _InquirePrompts()
                    return restart.inq_prompt()
                case "Exit/Quit":
                    sys.exit("Closing program...")
        
        # print(_stored_values._theta_0)
        func_prompts()

class FileReaders():
    """Container class for methods that read in and parse data files.
    """
    def _file_reader(cls, path : str, head_check : str):
        """Private method that is used to read data into a Pandas DataFrame

        Parameters
        ----------
        path : str
            File path for the desired data file
        
        head_check : str
            Header check for data files. Determines if header = None or 0.
            Accepts only 'y' | 'yes' and 'n' | 'no' in any capitalization.
            Any other passed in value defaults to 'yes'.

        Returns
        -------
        ndarray
            A NumPy array containing the data in the leftmost column of the
            passed in data file.
        
        ndarray
            A NumPy array containing the data in the second column of the
            passed in data file.
        """
        if path.endswith('csv'):
            print("CSV file")
        elif path.endswith('xlsx'):
            print('Excel file')
        # Converts the file into a pandas DataFrame
        match head_check.lower():
            case 'no' | 'n':
                header = None
            case 'yes' | 'y':
                header = 0
            case _:
                print("Invalid input, defaulting to 'yes'.")
                header = 0
        if path.endswith('csv'):
            datafile = pd.read_csv(path, header = header, index_col = None)
        elif path.endswith('xlsx'):
            datafile = pd.read_excel(path, header = header, index_col=None)
        
        # Saves column count for later use
        colcount = len(datafile.axes[1])
        
        # Converts the df into a numpy array
        dataarray = np.array(datafile)

        # Checks if the dimension of the array is equal to 1
        if np.ndim(dataarray) == 1:
            
            # Assumes given data is for the x_data and passes it into the x_data variable
            print("Data read into x_data")
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

    # External function that can be called by the user if they wish to. Is used only inside Data
    def csvreader(cls):
        """Reads in a csv file selected via a tkinter file explorer window.
        Assumes there is no index column. Data should be organized into columns
        rather than rows.

        Returns
        -------
        ndarray
            A NumPy array containing the data in the leftmost column of the
            passed in data file.
        
        ndarray
            A NumPy array containing the data in the second column of the
            passed in data file.
        """
        print("Please choose a csv file:")
        tk = Tk()
        tk.withdraw()
        
        # Opens a File Explorer window where the user can visually select a csv file
        path = askopenfilename(title="Select file", filetypes=(("CSV Files", '*.csv'),))
        header_check = input("Is there a header line? Y/N\n")
        # Prints the file path
        print('\n' + path)
        
        x_data, y_data = cls._file_reader(path=path, head_check=header_check)
        return x_data, y_data

    def excelreader(cls):
        """Reads in an Excel Workbook selected via a tkinter file explorer window.
        Assumes there is no index column. Data should be organized into columns
        rather than rows.

        Returns
        -------
        ndarray
            A NumPy array containing the data in the leftmost column of the
            passed in data file.
        
        ndarray
            A NumPy array containing the data in the second column of the
            passed in data file.
        """
        print("Please choose an Excel Workbook:")
        tk = Tk()
        tk.withdraw()
        
        # Opens a File Explorer window where the user can visually select a csv file
        path = askopenfilename(title="Select file", filetypes=(("Excel Workbook", '*.xlsx'),))
        header_check = input("Is there a header line? Y/N\n")
        print('\n')
        # Prints the file path
        print(path)
        
        x_data, y_data = cls._file_reader(path=path, head_check=header_check)
        return x_data, y_data

def _user_cli():
    """Initializes and calls the command-line-interface menus that can be used to
    make preset modifications and run functions.
    """
    inq = _InquirePrompts()
    inq.inq_prompt(inq)

if __name__ == "__main__":
    _user_cli()