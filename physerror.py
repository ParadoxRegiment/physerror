#       #####################################       #  
#        Module written by Alexander Ritzie         #
#       Originally created for BPHYS 312 AU23       #
#       at University of Washington - Bothell       #
#                   GitHub Repo:                    #
#    https://github.com/ParadoxRegiment/physerror   #
#                                                   #
#      Please direct any questions to my email:     #
#            alexander.ritzie@gmail.com             #
#       #####################################       #

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
import os

@dataclass
class Data():
    """ An initializer and container dataclass that is initialized and reused
    by the user and other classes, as well as their methods. There are many 
    attributes that calculate common statistical errors, constants, and 
    general error propagation methods, and one class Method to find, document, 
    and delete any data points that exist outside the standard 2 * sigma outlier
    "limit".
        
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
    user_x_data : np.ndarray
        []
    user_y_data : np.ndarray
        []
    
    Note
    ----
    Will update this soon-ish
    """
    user_x_data : ArrayLike = field(default_factory = lambda : [1,2,3,4,5])
    user_y_data : ArrayLike = field(default_factory = lambda : [1,2,3,4,5])
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
        returns that data back to the line where it was called. It is only
        used inside the __init__ function of this class.
            
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

        readcsv = input("Would you like to read in a CSV? Y/N\n")
        
        # Calls for the csvreader function if the user enters a yes(-adjecent) input
        if readcsv.lower() == "y" or readcsv.lower() == "yes":
            x_data, y_data = csvreader()
            
        # Assumes the user's passed in arguments are the desired data and passes them
        # into new variables                                                                             
        elif readcsv.lower() == "n" or readcsv.lower() == "no":
            if type(xdata) == np.ndarray:
                x_data = xdata
                y_data = ydata
            else:
                x_data = np.array(xdata)
                y_data = np.array(xdata)
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
        pulled from Data.
    """
    def __init__(cls):
        cls.graph_title = "Graph"
        cls.title_size = 11
        cls.x_label = "x label"
        cls.y_label = "y label"
        cls.p_color = 'cyan'
        cls.line_color = 'black'
        cls.errbar_color = 'red'
        cls.dist_check = 'No'
        cls.dataset_check = 1
        cls.hist_color = 'green'
    
    def linreg(cls, user_data : Data):
        """ Uses the given x_data and y_data arrays to create a linear
            regression plot.
            
            Parameters
            ----------
            user_data : Data
                Requires the user to pass in an instance of
                Data to make use of the user's data.
                    
            gtitle : str, optional
                The desired graph title. Defaults to "Graph".
            
            Returns
            -------
            plt.show()
                Opens an external window that shows the linear
                regression plot of the given data.
        """
        
        # New x_data and y_data for ease of use
        x_data = user_data._y_data                                
        y_data = user_data._x_data
        
        # Sets the figure's title to the default (or passed in) graph title
        plt.title(cls.graph_title, fontsize = cls.title_size)
        
        # Sets the figure data to x_data and y_data, colored orange
        plt.plot(x_data, y_data, 'o', color = cls.p_color)
        
        # Sets the figure's xlabel to the user's entered x_data name
        plt.xlabel(cls.x_label, fontsize = 11)
        
        # Sets the figure's xlabel to the user's entered y_data name
        plt.ylabel(cls.y_label, fontsize = 11)
        
        # Adds the linear regression line to the plot
        plt.plot(x_data, user_data.A + user_data.B * x_data)
        
        # Displays the linear regression plot
        plt.show()
    
    def errbargraph(cls, user_data : Data):
        """ Uses the given dataframe built from x_data and y_data during
            initalization to create an error bar plot, making use of
            the sigma_x value as the constant error.
            
            Parameters
            ----------
            user_data : Data
                Requires the user to pass in an instance of
                Data to make use of the user's data.
                
            gtitle : str, optional
                The desired graph title. Defaults to "Graph".
            
            Returns
            -------
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
                xlabel = cls.x_label, ylabel = cls.y_label, title = cls.graph_title,
                linestyle = "", marker = ".", yerr = user_data.sigma_x,
                capsize = 3, ecolor = cls.errbar_color, linewidth = 1)
        plt.show()
    
    def datahist(cls, user_data : Data):
        """ Uses the given dataframe built from x_data and y_data during
            initalization to create one or two histograms. There is also
            the option to turn the graphs into standard distribution
            graphs (currently a WIP).
        
            Parameters
            ----------
            user_data : Data
                Requires the user to pass in an instance of
                Data to make use of the user's data.
                
            gtitle : str, optional
                The desired graph title. Defaults to "Graph".
            
            Returns
            -------
            plt.show()
                Opens an external window that displays the
                histogram(s).
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
                plt.title(cls.graph_title)
                
            # Passes the if statement if the user entered a no(-adjecent) input
            elif stdcheck.lower() == "n" or stdcheck.lower() == "no":                                   
                pass
            else:
                # Runs the given print statement if anything other than yes or no is given
                print("Unknown input, assuming 'no'.")
                pass
        
        # Calls for and runs the histcheck function
        check = cls.dataset_check
        
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
                for i in range(check):
                    ax = axes[i]
                    
                    # Attaches the data from the first column of the df to the top plot                                      
                    datafile.hist(bins = len(datafile.axes[i]), grid = False, rwidth = .9,
                                            column = columns[i], color = cls.hist_color[i], ax = ax,
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
                
            gtitle : str, optional
                The desired graph title. Defaults to "Graph".
                
            marktype : str, optional
                The desired marker style. Defaults to "D".
                See link for all available matplotlib markers:
                https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
                
            markc : str, optional
                The desired marker color. Defaults to 'c'.
                See link for all matplotlob colors:
                https://matplotlib.org/stable/gallery/color/named_colors.html
                
            markedge : str, optional
                The desired marker edge color. Defaults to 'k'.
                See link for all matplotlob colors:
                https://matplotlib.org/stable/gallery/color/named_colors.html
            
            Returns
            -------
            plt.show()
                Opens an external window that displays the scatter plot.
        """
        
        # Local instances of user_data._x_data and user_data._y_data for ease of use
        x_data = user_data._x_data
        y_data = user_data._y_data
        
        # Sets a new pyplot figure with the 'constrained' layout
        plt.figure(num = 1, layout = 'constrained')
        
        # Generates a scatter plot using the given x_data for x and y_data for y
        # Optional customization options can be used to change the output graph
        plt.scatter(x = x_data, y = y_data, marker = "D", c = cls.p_color, edgecolors = 'k')
        
        # Pulls user_data colname1 and colname2 for the x-axis label and
        # y-axis label respectively.
        plt.xlabel(user_data._colname1)
        plt.ylabel(user_data._colname2)
        
        # Uses the passed in gtitle argument to set the plot's title
        plt.title(cls.graph_title)
        
        print("\nDisplaying graph using user's data...")
        
        # Displays the generated plot
        plt.show()
    
    def resid(cls, user_data: Data):
        """ Uses user_data._df to create a residuals scatter plot
            via the seaborn sns.residplot method. The graph's
            title can optionally be customized.

            Parameters
            ----------
            user_data : Data
                Requires the user to pass in an instance of
                Data to make use of the user's data.
                
            gtitle : str, optional
                The desired graph title. Defaults to "Graph".
            
            Returns
            -------
            plt.show()
                Opens an external window that displays the residuals scatter plot.
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
        
    def dbl_pend(theta_0 : float, phi_0 : float, theta_dot_0 = 0, phi_dot_0 = 0, anim_type = 0):
        import matplotlib.animation as animation
        gravity = 9.81      # m/s^2
        
        print('Program defaults to generating a point-mass double pendulum animation.\n'
              'If you want a bar-mass double pendulum animation, make sure anim_type = 1\n')
        
        len_1 = float(input('Enter the length of the top pendulum in meters: '))
        len_2 = float(input('Enter the length of the bottom pendulum in meters: '))
        tot_len = len_1 + len_2
        mass_1 = float(input('Enter the mass of the top pendulum in kg: '))
        mass_2 = float(input('Enter the mass of the bottom pendulum in kg: '))
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

        line, = ax.plot([], [], 'o-', color='lime', lw=2)
        trace, = ax.plot([], [], 'k.-', lw=1, ms=2)
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


        ani = animation.FuncAnimation(fig, animate, len(output), interval=dt*1000, blit=True)
        ani.save(f'{theta_0}{phi_0}{anim_type}anim.gif', writer='imagemagick', fps=20)
        plt.show()

class _InquirePrompts:
    def __init__(cls):
        cls.graphs_obj = Graphs()
        
        cls._title_prompt = f"Title - {cls.graphs_obj.graph_title}"
        cls._title_size_prompt = f"Title Size - {cls.graphs_obj.title_size}"
        cls._x_label_prompt = f"x-label - {cls.graphs_obj.x_label}"
        cls._y_label_prompt = f"y-label - {cls.graphs_obj.y_label}"
        cls._point_colors_prompt = f"Point Colors - {cls.graphs_obj.p_color}"
        cls._line_color_prompt = f"Line Color - {cls.graphs_obj.line_color}"
        cls._errbar_color_prompt = f"Error Bar Color - {cls.graphs_obj.errbar_color}"
        cls._hist_color_prompt = f"Histogram Color(s) - {cls.graphs_obj.hist_color}"
        
        cls.data_q = [
                inquirer.List(
                    "data",
                    message="Select a data file type",
                    choices=["CSV", "Excel", "XML - WIP", "JSON - WIP"],
                    ),
            ]
        cls.funcs_q = [
            inquirer.List(
                "function",
                message="Select a function to modify and/or run",
                choices=["Data.outlier",
                         "Graphs.linreg",
                         "Graphs.errbargraph",
                         "Graphs.datahist",
                         "Graphs.sctrplot",
                         "Graphs.resid",
                         "Graphs.dbl_pend",
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
        
    def _inq_prompt(cls):
        data_ans = inquirer.prompt(cls.data_q)
        tempdata = Data()
        # "Title", "Title Size", "x-label", "y-label", "Point Color(s)", "Line Color", "Run", "Back"
        def linreg_prompts(graph_obj):
            match inquirer.prompt(cls.linreg_q)["linreg"]:
                case "Title":
                    graph_obj.graph_title = input("Enter a graph title: ")
                    # print(cls.graphs_obj.graph_title)
                    
                    return linreg_prompts()
                case "Run":
                    cls.graphs_obj.linreg(tempdata)
                case "Back":
                    func_prompts()
                case "Exit/Quit":
                    sys.exit()
                case _:
                    print('WIP section')
        
        def errbar_prompts():
            match inquirer.prompt(cls.errbar_q)["errbar"]:
                case cls._title_prompt:
                    cls.graphs_obj.graph_title = input("Enter a graph title: ")
                    return errbar_prompts()
                case cls._title_size_prompt:
                    print("WIP")
                case cls._x_label_prompt:
                    print("WIP")
                case cls._y_label_prompt:
                    print("WIP")
                case cls._point_colors_prompt:
                    print("WIP")
                case cls._errbar_color_prompt:
                    print("WIP")
                case "Run":
                    cls.graphs_obj.errbargraph(tempdata)
                case "Back":
                    func_prompts()
                case "Exit/Quit":
                    sys.exit()
        
        # "Title", "Title Size", "Normal Distribution", "Bar Color", "Dataset Count", "Run", "Back"
        def datahist_prompts():
            match inquirer.prompt(cls.datahist_q)["datahist"]:
                case "Title":
                    cls.graphs_obj.graph_title = input("Enter a new graph title: ")
                    return datahist_prompts()
                case "Title Size":
                    try:
                        cls.graphs_obj.title_size = float(input("Enter a new title size: "))
                    except ValueError as e:
                        e.add_note("Only numerical values are accepted")
                        return datahist_prompts()
                case "x-label":
                    print("WIP")
                case "y-label":
                    print("WIP")
                case "Point Color(s)":
                    print("WIP")
                case "Line Color":
                    print("WIP")
                case "Run":
                    cls.graphs_obj.errbargraph(tempdata)
                case "Back":
                    func_prompts()
                case "Exit/Quit":
                    sys.exit()
        
        # "Title", "Title Size", "x-label", "y-label", "Point Color(s)", "Run", "Back"
        def sctrplot_prompts():
            return
        
        # "Title", "Title Size", "x-label", "y-label", "Point Color(s)", "Line Color", "Run", "Back"
        def resid_prompts():
            return
        
        def dbl_pend_prompts():
            return
        
        def func_prompts():
            funcs_ans = inquirer.prompt(cls.funcs_q)
            match funcs_ans["function"]:
                case "Data.outlier":
                    cls.graphs_obj.graph_title = inquirer.prompt(cls.gtitle_q)["title"]
                    print(cls.graphs_obj.graph_title)
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
                case "Exit/Quit":
                    sys.exit()
            
        func_prompts()
    
def user_cli():
    inq = _InquirePrompts()
    inq._inq_prompt()

# External function that can be called by the user if they wish to. Is used only inside Data
def csvreader()-> np.ndarray:
    print("Please choose a csv file:")
    tk = Tk()
    tk.withdraw()
    
    # Opens a File Explorer window where the user can visually select a csv file
    path = askopenfilename(title="Select file", filetypes=(("CSV Files", '*.csv'),))
    
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

if __name__ == "__main__":
    import sys
    import subprocess
    import inquirer

    def install(package):
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

    def check_install():
        reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
        installed_packages = [r.decode().split('==')[0] for r in reqs.split()]
        
        return installed_packages

    
    req_packages = ['numpy', 'pandas', 'matplotlib', 'tk', 'scipy', 'seaborn', 'inquirer']
    installed_packages = check_install()
    print(f"Checking for required packages: {req_packages}\n")
    
    for pkg in req_packages:
        if pkg in installed_packages:
            print(f"{pkg} is already installed, continuing...")
            continue
        else:
            print(f"Installing {pkg}...\n")
            install(pkg)
    
    print("Required packages check complete.")
    installed_packages = check_install()
    print(f"\nInstalled packages w/ dependencies:\n{installed_packages}\n")
    
    user_cli()