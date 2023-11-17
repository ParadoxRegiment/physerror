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
    def __init__(self):
        self.x_data = np.array([])
        self.y_data = np.array([])
        self.df = pd.DataFrame()
        self.colname1 = ''
        self.colname2 = ''

class DataInit:
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
        userdata.colname1 = input("Please type first data set's name: ")
        userdata.colname2 = input("Please type second data set's name: ")
        datafile = pd.DataFrame(np.transpose(temparray), columns = [userdata.colname1, userdata.colname2])
        datafile.index.name = 'Trial'
        datafile.index += 1
        
        print(datafile)

        userdata.df = datafile
        userdata.x_data = x_data
        userdata.y_data = y_data
        self.colname1 = userdata.colname1
        self.colname2 = userdata.colname2

class calcs:
    """
    To access attributes, must do [variable] = calcs(ndarray, ndarray)
    To access methods, do [above variable].[function name]
    
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
        """
        Initalization function. x_data and y_data are initalized to
        arrays of size 5 filled with 5s.
        
        For csv data files, please make sure your x data is in the first
        row, and y data is in the second row.
        
        Parameters:
            x_data : ndarray [optional]
                The desired x_data array. Initalized to np.full(5,5)
            y_data : ndarray [optional]
                The desired y_data array. Initalized to np.full(5,5)
        """
        
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
        """
            Checks for and prints out any outliers in included arrays within the 2 * sigma limit.
            
            Returns: print()
                        Prints out x_outliers and y_outliers in the form of ndarrays
        """
        x_data = self.x_data
        y_data = self.y_data
        
        print("Given x data:", x_data)
        print("Given y data:", y_data)
        
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

        print(x_outliers)
        print(y_outliers)

class Graphs:
    def __init__(self):
        global calcsclass
        calcsclass = calcs()
    
    def linregress(self):
        """
            Uses the given x_data and y_data arrays to create a linear regression plot.
            
            Returns: plt.show()
                        Opens an external window that shows the linear regression plot of the given data. 
        """
        x_data = userdata.x_data
        y_data = userdata.y_data
        
        if np.array_equiv(testarray, y_data):
            y_data = np.full_like(x_data, 5)
        
        plt.title('pyplot best fit & linear regression plot', fontsize = 14)
        plt.plot(x_data, y_data, 'o')
        plt.xlabel(userdata.colname1, fontsize = 14)
        plt.ylabel(userdata.colname2, fontsize = 14)
        plt.plot(x_data, calcsclass.A + calcsclass.B * x_data)
        plt.show()
        
    def standdistgraph(self):
        """
            Uses the given x_data array to create a standard distribution graph.
            Currently only works for x data.
                
            Returns: plt.show()
                        Opens an external window that shows the standard distribution graph
        """
        x_data = userdata.x_data
        y_data = userdata.y_data
        
        gdata = x_data

        plt.hist(gdata, bins = len(gdata), density = True, alpha = 0.6, color = 'c', align = 'mid')
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, calcsclass.x_mean, calcsclass.sigma_x)
        plt.plot(x, p, 'k', linewidth = 2)
        title = "x data mean = %.2f, stddev x = %.2f, 2 stddev x = %.2f" % (calcsclass.x_mean, calcsclass.sigma_x, 2 * calcsclass.sigma_x)
        plt.title(title)
        plt.show()
    
    def errbargraph(self, x_axis = "y_data", y_axis = "x_data", gtitle = "graph"):
        """
            Uses the given dataframe built from x_data and y_data during
            initalization to create an error bar plot, making use of
            the sigma_x value as the constant error.
            
            Parameters:
                x_axis : str [optional]
                    The desired label for the x axis. Is initalized to
                    "y_data" (as most datasets will have the y data
                    in the second column)
                y_axis : str [optional]
                    The desired label for the y axis. Is initalized to
                    "x_data" (as most data sets will have the x data
                    in the first column)
                gtitle : str [optional]
                    The desired graph title. is initalized to "graph"
            
            Returns: plt.show()
                        Opens an external window that displays the
                        error bar graph.
        """
        # df = data.df.set_axis(['X', 'Y'], axis = 1)
        df = userdata.df
        df.plot(x = userdata.colname2, y = userdata.colname1,
                xlabel = x_axis, ylabel = y_axis, title = gtitle,
                linestyle = "", marker = ".", yerr = calcsclass.sigma_x,
                capthick = 1, ecolor = "red", linewidth = 1)
        plt.show()
    
    def datahist(self):
        datafile = userdata.df
        columns = datafile.columns
        histcheck = input("Do you want a histogram for 1 dataset, or 2 datasets? ")
        
        try:
            int(histcheck)
        except ValueError:
            print("Please input only numerical values 1 or 2.")
            exit()
            
        if int(histcheck) == 1:
            print("Which dataset would you like to use?", userdata.colname1, "or", userdata.colname2,)
            histnum = input()
            if histnum == userdata.colname1:
                datafile.hist(bins = len(datafile.axes[0]), grid = False, rwidth = .9,
                                         column = columns[0], color = 'green')
            elif histnum == userdata.colname2:
                datafile.hist(bins = len(datafile.axes[0]), grid = False, rwidth = .9,
                                         column = columns[1], color = 'green')
            else:
                print("Please input only", userdata.colname1, "or", userdata.colname2)
                exit()
        elif int(histcheck) == 2:
            fig, axes = plt.subplots(nrows = 2, ncols = 1)
            datafile.hist(bins = len(datafile.axes[0]), grid = False, rwidth = .9,
                                     column = columns[0], color = 'green', ax = axes[0])
            datafile.hist(bins = len(datafile.axes[1]), grid = False, rwidth = .9,
                                     column = columns[1], color = 'c', ax = axes[1])
        else:
            print("Please input only 1 or 2.")
            exit
        plt.show()
        
    # def nonlinregress(data):
    #     df = data.df
    #     # container = np.array([])
    #     def f_model(x, a, c):
    #         return np.log(np.array(((a + x) ** 2) / ((x - c) ** 2)))
