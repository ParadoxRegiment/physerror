#   #####################################   #  
#    Module written by Alexander Ritzie     #
#   Originally created for BPHYS 231 AU23   #
#   at University of Washington - Bothell   #
#   #####################################   #
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pandas as pd

class Calculations:
    """
    To access array values and functions do:
    [variable] = Calculuations(ndarray[optional], ndarray[optional])
    
    To access methods, do:
    [above variable].[method name]

    -----------------------------------------------
    
    Available array values:
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
    
    def __init__(data, x_data = np.full(5,5), y_data = np.full(5,5)):
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
        
        
        readcsv = input("Would you like to read in a CSV? Y/N\n")
        if readcsv.lower() == "y" or readcsv.lower() == "yes":
            print("Please choose a csv file:")
            tk = Tk()
            tk.withdraw()
            path = askopenfilename()
            print(path)
            with open(path, "r") as f:
                datafile = np.array(pd.read_csv(f))
            
            datafile = np.delete(datafile, 0, 1)

            if np.ndim(datafile) == 1:
                print("csv read into x_data")
                x_data = datafile
            elif np.ndim(datafile) > 1:
                x_data = datafile[0]
                y_data = datafile[1]
    
        elif readcsv.lower() == "n" or readcsv.lower() == "no":
            pass
        else:
            print("Unknown input, please restart.")
            exit
        
        global testarray
        testarray = np.full(5,5)
        
        if np.array_equiv(testarray, y_data):
            y_data = np.full_like(x_data, 5)

        data.x_data = x_data
        data.y_data = y_data
        data.N = len(x_data)
        data.delta = data.N * sum(x_data ** 2) - (sum(x_data)) ** 2
        data.A = ((sum(x_data ** 2) * sum(y_data)) - (sum(x_data) * sum(x_data * y_data))) / data.delta
        data.B = (data.N * sum(x_data * y_data) - (sum(x_data) * sum(y_data))) / data.delta
        data.x_mean = abs(np.mean(x_data))
        data.x_best = sum(x_data)/data.N
        data.y_mean = abs(np.mean(y_data))
        data.y_best = sum(y_data)/data.N
        data.sigma_y = np.sqrt((1/(data.N - 2)) * sum((y_data - data.A - (data.B * x_data)) ** 2))
        data.sigma_A = data.sigma_y * np.sqrt(sum(x_data ** 2) / data.delta)
        data.sigma_B = data.sigma_y * np.sqrt(data.N / data.delta)
        data.sigma_x = np.sqrt(sum((x_data - data.x_mean) ** 2) / (data.N - 1))
        data.sigma_x_best = np.sqrt((1/(data.N - 1)) * sum((x_data - data.x_mean) ** 2))
        data.sigma_y_best = np.sqrt((1/(data.N - 1)) * sum((y_data - data.y_mean) ** 2))
        data.sigma_x_mean = data.x_mean / np.sqrt(data.N)
        data.sigma_y_mean = data.y_mean / np.sqrt(data.N)
        data.sigma_frac = 1 / np.sqrt(2 * (data.N - 1))
    
    def outlier(data):
        """
            Checks for and prints out any outliers in included arrays within the 2 * sigma limit.
            
            Returns: print()
                        Prints out x_outliers and y_outliers in the form of ndarrays
        """
        x_data = data.x_data
        y_data = data.y_data
        
        if np.array_equiv(testarray, y_data):
            y_data = np.full_like(x_data, 5)
        
        print("Given x data:", x_data)
        print("Given y data:", y_data)
        
        x_outliers = np.zeros(len(x_data))
        y_outliers = np.zeros(len(y_data))
        i = 0
        j = 0
        k = 0
        l = 0
        
        for row in x_data:
            if row > (data.x_mean + 2 * data.sigma_x):
                x_outliers[j] = int(row)
                j += 1
                x_data = np.delete(x_data, i)
            elif row < (data.x_mean - 2 * data.sigma_x):
                x_outliers[j] = int(row)
                j += 1
                x_data = np.delete(x_data, i)
            i += 1
            
        for row in y_data:
            if row > (data.y_mean + 2 * data.sigma_y):
                y_outliers[l] = int(row)
                l += 1
                y_data = np.delete(y_data, k)
            elif row < (data.y_mean - 2 * data.sigma_y):
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
        
    def regress(data):
        """
            Uses the given x_data and y_data arrays to create a linear regression plot.
            
            Returns: plt.show()
                        Opens an external window that shows the linear regression plot of the given data. 
        """
        x_data = data.x_data
        y_data = data.y_data
        
        if np.array_equiv(testarray, y_data):
            y_data = np.full_like(x_data, 5)
        
        print("Given x data:", x_data)
        print("Given y data:", y_data)
        
        plt.title('pyplot best fit & linear regression plot')
        plt.plot(x_data, y_data, 'o')
        plt.plot(x_data, data.A + data.B * x_data)
        plt.show()
    
    def standdistgraph(data):
        """
            Uses the given x_data array to create a standard distribution graph.
            Currently only works for x data.
                
            Returns: plt.show()
                        Opens an external window that shows the standard distribution graph
        """
        x_data = data.x_data
        y_data = data.y_data
        
        gdata = x_data

        plt.hist(gdata, bins = len(gdata), density = True, alpha = 0.6, color = 'c', align = 'mid')
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, data.x_mean, data.sigma_x)
        plt.plot(x, p, 'k', linewidth = 2)
        title = "x data mean = %.2f, stddev x = %.2f, 2 stddev x = %.2f" % (data.x_mean, data.sigma_x, 2 * data.sigma_x)
        plt.title(title)
        plt.show()
        
