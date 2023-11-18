#       #####################################       #  
#        Module written by Alexander Ritzie         #
#       Originally created for BPHYS 312 AU23       #
#       at University of Washington - Bothell       #
#                   GitHub Repo:                    #
#    https://github.com/ParadoxRegiment/BPHYS231    #
#       #####################################       #
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pandas as pd

class Data:
    """
    A container class that cannot be initialized directly.
    It contains the user's x_data and y_data arrays, the
    pandas DataFrame created from those arrays, and the
    data name (headers) given by the user.
    """
    def __init__(self):
        self.x_data = np.array([])
        self.y_data = np.array([])
        self.df = pd.DataFrame()
        self.colname1 = ""
        self.colname2 = ""

class DataInit:
    """
    Initalization class that reads in data either directly
    from the program or from an external csv file. The
    user is also asked to name their data.
    
    Note: Although csv files of any shape can be read in,
    it is recommended to use csv files with two columns of
    data without a header row or index column.
        * This is because this module was created for students
        that may not be well versed in programming or data
        analysis.
    
    Parameters:
        x_data : ndarray [optional]
            An optional parameter that allows the user to
            read in data from within their program itself.
            x_data is initalized to np.full(5,5)
        
        y_data : ndarray [optional]
            An optional parameter that allows the user to
            read in data from within their program itself.
            y_data is initalized to np.full(5,5)
            
    Returns:
        None
    """
    
    def __init__(self, x_data = np.full(5,5), y_data = np.full(5,5)):
        global userdata
        userdata = Data()
        readcsv = input("Would you like to read in a CSV? Y/N\n")
        if readcsv.lower() == "y" or readcsv.lower() == "yes":
            print("Please choose a csv file:")
            tk = Tk()
            tk.withdraw()
            path = askopenfilename()
            print(path)
            with open(path, "r") as f:
                datafile = pd.read_csv(f, header = None, index_col = None)
                colcount = len(datafile.axes[1])
                dataarray = np.array(datafile)

            if np.ndim(dataarray) == 1:
                print("csv read into x_data")
                x_data = dataarray
            elif np.ndim(dataarray) > 1:
                if colcount == 1:
                    print("csv read into x_data")
                    x_data = dataarray[:,0]
                elif colcount > 1:
                    x_data = dataarray[:,0]
                    y_data = dataarray[:,1]
        elif readcsv.lower() == "n" or readcsv.lower() == "no":
            pass
        else:
            print("Unknown input, please restart.")
            exit()
        
        global testarray
        testarray = np.full(5,5)
        
        if np.array_equiv(testarray, y_data):
            y_data = np.full_like(x_data, 5)
        
        temparray = np.stack((x_data, y_data))
        self.colname1 = input("Please type first data set's name: ")
        self.colname2 = input("Please type second data set's name: ")
        datafile = pd.DataFrame(np.transpose(temparray), columns = [self.colname1, self.colname2])
        datafile.index.name = 'Trial'
        datafile.index += 1
        
        print(datafile)

        userdata.df = datafile
        userdata.x_data = x_data
        userdata.y_data = y_data
        userdata.colname1 = self.colname1
        userdata.colname2 = self.colname2

class calcs:
    """
    Available methods:
        outlier:
            Checks for and prints out any outliers in included arrays
            within the 2 * sigma limit.
            
            Parameters: None
            
            Returns: print()
                        Prints out x_outliers and y_outliers in the
                        form of ndarrays

    Available attributes:
        N: size of x data
        delta: the calculated delta constant
        A: the calculated A constant
        B: the calculated B constant
        x_mean: the calculated x data mean
        x_best: best estimated value of x data
        y_mean: the calculated y data mean
        y_best: best estimated value of y data
        sigma_x: the calculated sigma of x data
        sigma_y: the calculated sigma of y data
        sigma_A: the calculated sigma of constant A
        sigma_B: the calculated sigma of constant B
        sigma_x_best: the calculated best sigma of x data
        sigma_y_best: the calculated best sigma of y data
        sigma_x_mean: the calculated sigma of the x data mean
        sigma_y_mean: the calculated sigma of the y data mean
        sigma_frac: the calculated fractional uncertainty
    """
    def __init__(self):        
        self.df = userdata.df
        self.x_data = userdata.x_data
        self.y_data = userdata.y_data
        self.N = len(userdata.x_data)
        self.delta = self.N * sum(userdata.x_data ** 2) - (sum(userdata.x_data)) ** 2
        self.A = ((sum(userdata.x_data ** 2) * sum(userdata.y_data)) - (sum(userdata.x_data) * sum(userdata.x_data * userdata.y_data))) / self.delta
        self.B = (self.N * sum(userdata.x_data * userdata.y_data) - (sum(userdata.x_data) * sum(userdata.y_data))) / self.delta
        self.x_mean = abs(np.mean(userdata.x_data))
        self.x_best = sum(userdata.x_data)/self.N
        self.y_mean = abs(np.mean(userdata.y_data))
        self.y_best = sum(userdata.y_data)/self.N
        self.sigma_y = np.sqrt((1/(self.N - 2)) * sum((userdata.y_data - self.A - (self.B * userdata.x_data)) ** 2))
        self.sigma_A = self.sigma_y * np.sqrt(sum(userdata.x_data ** 2) / self.delta)
        self.sigma_B = self.sigma_y * np.sqrt(self.N / self.delta)
        self.sigma_x = np.sqrt(sum((userdata.x_data - self.x_mean) ** 2) / (self.N - 1))
        self.sigma_x_best = np.sqrt((1/(self.N - 1)) * sum((userdata.x_data - self.x_mean) ** 2))
        self.sigma_y_best = np.sqrt((1/(self.N - 1)) * sum((userdata.y_data - self.y_mean) ** 2))
        self.sigma_x_mean = self.x_mean / np.sqrt(self.N)
        self.sigma_y_mean = self.y_mean / np.sqrt(self.N)
        self.sigma_frac = 1 / np.sqrt(2 * (self.N - 1))
    
    def outlier(self):
        x_data = self.x_data
        y_data = self.y_data
        
        x_outliers = np.zeros(len(x_data))
        y_outliers = np.zeros(len(y_data))
        i = 0
        j = 0
        k = 0
        l = 0
        
        for row in x_data:
            if row > (self.x_mean + 2 * self.sigma_x):
                x_outliers[j] = int(row)
                j += 1
                x_data = np.delete(x_data, i)
            elif row < (self.x_mean - 2 * self.sigma_x):
                x_outliers[j] = int(row)
                j += 1
                x_data = np.delete(x_data, i)
            i += 1
            
        for row in y_data:
            if row > (self.y_mean + 2 * self.sigma_y):
                y_outliers[l] = int(row)
                l += 1
                y_data = np.delete(y_data, k)
            elif row < (self.y_mean - 2 * self.sigma_y):
                y_outliers[l] = int(row)
                l += 1
                y_data = np.delete(y_data, k)
            k += 1
            
        x_outliers.resize(j)
        y_outliers.resize(l)
        
        if np.size(x_outliers) == 0:
            x_outliers = 'No outliers in x data'
        
        if np.size(y_outliers) == 0:
            y_outliers = 'No outliers in y data'

        print(userdata.colname1, "outliers:", x_outliers)
        print(userdata.colname2, "outliers:", y_outliers)

class Graphs:
    """
    Allows the user to create various graphs from the data
    read into DataInit().
    
    Available methods:
        regress:
            Uses the given x_data and y_data arrays to create a linear
            regression plot.
            
            Parameters:
                gtitle : str [optional]
                    The desired graph title. is initalized to "graph"
            
            Returns:
                plt.show()
                    Opens an external window that shows the linear
                    regression plot of the given data.
        
        errbargraph:
            Uses the given dataframe built from x_data and y_data during
            initalization to create an error bar plot, making use of
            the sigma_x value as the constant error.
            
            Parameters:
                gtitle : str [optional]
                    The desired graph title. is initalized to "graph"
            
            Returns:
                plt.show()
                    Opens an external window that displays the
                    error bar graph.
                        
        datahist:
            Uses the given dataframe built from x_data and y_data during
            initalization to create one or two histograms. There is also
            the option to turn the graphs into standard distribution
            graphs (currently a WIP).
            
            Parameters:
                gtitle : str [optional]
                    The desired graph title. is initalized to "graph"
            
            Returns:
                plt.show()
                    Opens an external window that displays the
                    histogram(s).
    """
    
    def __init__(self):
        global calcsclass
        calcsclass = calcs()
    
    def regress(self, gtitle = "graph"):
        x_data = userdata.x_data
        y_data = userdata.y_data
        
        if np.array_equiv(testarray, y_data):
            y_data = np.full_like(x_data, 5)
        
        plt.title(gtitle, fontsize = 11)
        plt.plot(x_data, y_data, 'o')
        plt.xlabel(userdata.colname1, fontsize = 11)
        plt.ylabel(userdata.colname2, fontsize = 11)
        plt.plot(x_data, calcsclass.A + calcsclass.B * x_data)
        plt.title(gtitle)
        plt.show()
        
    ##### This method is currently being integrated into datahist
    ##### and will be officially removed upon completion.
    # def standdistgraph(self, gtitle = "graph"):
    #     x_data = userdata.x_data
    #     y_data = userdata.y_data
        
    #     gdata = x_data

    #     plt.hist(gdata, bins = len(gdata), density = True, alpha = 0.6,
    #              color = 'c', align = 'mid')
    #     xmin, xmax = plt.xlim()
    #     x = np.linspace(xmin, xmax, 100)
    #     p = stats.norm.pdf(x, calcsclass.x_mean, calcsclass.sigma_x)
    #     plt.plot(x, p, 'k', linewidth = 2)
    #     plt.title(gtitle)
    #     plt.show()
    
    def errbargraph(self, gtitle = "graph"):
        df = userdata.df
        df.plot(x = userdata.colname2, y = userdata.colname1,
                xlabel = userdata.colname2, ylabel = userdata.colname1, title = gtitle,
                linestyle = "", marker = ".", yerr = calcsclass.sigma_x,
                capthick = 1, ecolor = "red", linewidth = 1)
        plt.show()
    
    def datahist(self, gtitle = "graph"):
        datafile = userdata.df
        columns = datafile.columns
        
        stdcheck = input("Will this a standard distribution graph? Y/N ")
        
        def stdcheckfunc():
            if stdcheck.lower() == "y" or stdcheck.lower() == "yes":
                xmin, xmax = plt.xlim()
                x = np.linspace(xmin, xmax, 100)
                p = stats.norm.pdf(x, calcsclass.x_mean, calcsclass.sigma_x)
                plt.plot(x, p, 'k', linewidth = 2)
                plt.title(gtitle)
            elif stdcheck.lower() == "n" or stdcheck.lower() == "no":
                pass
            else:
                print("Unknown input, assuming 'no'.")
        
        histcheck = input("Do you want a histogram for 1 dataset, or 2 datasets? ")
        
        try:
            int(histcheck)
        except ValueError:
            print("Please input only numerical values 1 or 2.")
            exit()
            
            # For this entire method I'm not quite sure how to get both histograms to show the
            # gaussian line when the user chooses to show both histograms at once.
        if int(histcheck) == 1:
            print("Which dataset would you like to use?", userdata.colname1, "or", userdata.colname2,)
            histnum = input()
            if histnum == userdata.colname1:
                datafile.hist(bins = len(datafile.axes[0]), grid = False, rwidth = .9,
                                         column = columns[0], color = 'green', density = True)
                stdcheckfunc()
            elif histnum == userdata.colname2:
                datafile.hist(bins = len(datafile.axes[0]), grid = False, rwidth = .9,
                                         column = columns[1], color = 'green', density = True)
                stdcheckfunc()
            else:
                print("Please input only", userdata.colname1, "or", userdata.colname2)
                exit()
        elif int(histcheck) == 2:
            fig, axes = plt.subplots(nrows = 2, ncols = 1)
            datafile.hist(bins = len(datafile.axes[0]), grid = False, rwidth = .9,
                                     column = columns[0], color = 'green', ax = axes[0],
                                     density = True)
            stdcheckfunc()
            datafile.hist(bins = len(datafile.axes[1]), grid = False, rwidth = .9,
                                     column = columns[1], color = 'c', ax = axes[1],
                                     density = True)
            stdcheckfunc()
        else:
            print("Please input only 1 or 2.")
            exit

        plt.show()
        
    # def nonlinregress(data):
    #     df = data.df
    #     # container = np.array([])
    #     def f_model(x, a, c):
    #         return np.log(np.array(((a + x) ** 2) / ((x - c) ** 2)))
