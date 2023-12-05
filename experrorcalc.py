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
from dataclasses import dataclass
from typing import Type

@dataclass
class Data():
    """
    A container class that cannot be initialized directly.
    It contains the user's x_data and y_data arrays, the
    pandas DataFrame created from those arrays, and the
    data name (headers) given by the user.
    """
    def __init__(self, user_x_data = np.array([]), user_y_data = np.array([])):
        self.x_data, self.y_data, self.df, self.colname1, self.colname2 = self.initfunc(user_x_data, user_y_data)
        
        if len(self.x_data) != 0:
            N = len(self.x_data)
            self.delta = N * sum(self.x_data ** 2) - (sum(self.x_data)) ** 2
            self.A = ((sum(self.x_data ** 2) * sum(self.y_data)) - (sum(self.x_data) * sum(self.x_data * self.y_data))) / self.delta
            self.B = (N * sum(self.x_data * self.y_data) - (sum(self.x_data) * sum(self.y_data))) / self.delta
            self.x_mean = abs(np.mean(self.x_data))
            self.x_best = sum(self.x_data)/N
            self.y_mean = abs(np.mean(self.y_data))
            self.y_best = sum(self.y_data)/N
            self.sigma_y = np.sqrt((1/(N - 2)) * sum((self.y_data - self.A - (self.B * self.x_data)) ** 2))
            self.sigma_A = self.sigma_y * np.sqrt(sum(self.x_data ** 2) / self.delta)
            self.sigma_B = self.sigma_y * np.sqrt(N / self.delta)
            self.sigma_x = np.sqrt(sum((self.x_data - self.x_mean) ** 2) / (N - 1))
            self.sigma_x_best = np.sqrt((1/(N - 1)) * sum((self.x_data - self.x_mean) ** 2))
            self.sigma_y_best = np.sqrt((1/(N - 1)) * sum((self.y_data - self.y_mean) ** 2))
            self.sigma_x_mean = self.x_mean / np.sqrt(N)
            self.sigma_y_mean = self.y_mean / np.sqrt(N)
            self.sigma_frac = 1 / np.sqrt(2 * (N - 1))
    
    def initfunc(self, xdata = np.full(5,5), ydata = np.full(5,5)):
        readcsv = input("Would you like to read in a CSV? Y/N\n")
        if readcsv.lower() == "y" or readcsv.lower() == "yes":
            x_data, y_data = csvreader(xdata, ydata)
        elif readcsv.lower() == "n" or readcsv.lower() == "no":
            x_data = xdata
            y_data = ydata
            pass
        else:
            print("Unknown input, please restart.")
            exit()
        
        testarray = np.full(5,5)
            
        if np.array_equiv(testarray, y_data):
            y_data = np.full_like(x_data, 5)
        
        temparray = np.stack((x_data, y_data))
        colname1 = input("Please type the first data set's name: ")
        colname2 = input("Please type the second data set's name: ")
        datafile = pd.DataFrame(np.transpose(temparray), columns = [colname1, colname2])
        datafile.index.name = 'Trial'
        datafile.index += 1
        del temparray, testarray
        
        print(datafile)
        
        return x_data, y_data, datafile, colname1, colname2
    
    def outlier(self):
        x_data = self.x_data
        y_data = self.y_data
        
        if np.size(x_data) == 0 or np.size(y_data) == 0:
            exit
        
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

        return x_outliers, y_outliers

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
    
    def regress(USERDATA : Type[Data], gtitle = "graph"):
        x_data = USERDATA.x_data
        y_data = USERDATA.y_data
        
        plt.title(gtitle, fontsize = 11)
        plt.plot(x_data, y_data, 'o')
        plt.xlabel(USERDATA.colname1, fontsize = 11)
        plt.ylabel(USERDATA.colname2, fontsize = 11)
        plt.plot(x_data, USERDATA.A + USERDATA.B * x_data)
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
    
    def errbargraph(USERDATA : Type[Data], gtitle = "graph"):
        df = USERDATA.df
        df.plot(x = USERDATA.colname2, y = USERDATA.colname1,
                xlabel = USERDATA.colname2, ylabel = USERDATA.colname1, title = gtitle,
                linestyle = "", marker = ".", yerr = USERDATA.sigma_x,
                capthick = 1, ecolor = "red", linewidth = 1)
        plt.show()
    
    def datahist(USERDATA : Type[Data], gtitle = "graph"):
        datafile = USERDATA.df
        columns = datafile.columns
        
        stdcheck = input("Will this a standard distribution graph? Y/N ")
        
        def stdcheckfunc():
            if stdcheck.lower() == "y" or stdcheck.lower() == "yes":
                xmin, xmax = plt.xlim()
                x = np.linspace(xmin, xmax, 100)
                p = stats.norm.pdf(x, USERDATA.x_mean, USERDATA.sigma_x)
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
            print("Which dataset would you like to use?", USERDATA.colname1, "or", USERDATA.colname2,)
            histnum = input()
            if histnum == USERDATA.colname1:
                datafile.hist(bins = len(datafile.axes[0]), grid = False, rwidth = .9,
                                         column = columns[0], color = 'green', density = True)
                stdcheckfunc()
            elif histnum == USERDATA.colname2:
                datafile.hist(bins = len(datafile.axes[0]), grid = False, rwidth = .9,
                                         column = columns[1], color = 'green', density = True)
                stdcheckfunc()
            else:
                print("Please input only", USERDATA.colname1, "or", USERDATA.colname2)
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
            exit()

        plt.show()
    # def nonlinregress(data):
    #     df = data.df
    #     # container = np.array([])
    #     def f_model(x, a, c):
    #         return np.log(np.array(((a + x) ** 2) / ((x - c) ** 2)))

def csvreader(x_data : np.array, y_data : np.array)-> np.array:
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
    return x_data, y_data
