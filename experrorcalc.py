import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats

class Calculations:
    """
    To access array values and functions, must do [variable] = Calculuations(ndarray, ndarray)
    To access functions, do [above variable].[function name]
    
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
    
    
    def __init__(data, x_data = np.full(5, 5), y_data = np.full(5, 5)):
        
        global testarray
        testarray = np.full(5,5)
        
        if np.array_equiv(testarray, y_data):
            y_data = np.full_like(x_data, 5)
        
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
    
    def outlier(data, x_data = np.zeros(0), y_data = np.zeros(0)):
        """
            Checks for and prints out any outliers in included arrays within the 2 * sigma limit.
            
            Parameters:
                x_data  :   1darray
                    The desired x_data array
                y_data  :   1darray
                    The desired y_data array
            
            Returns: print()
                        Prints out x_outliers and y_outliers in the form of ndarrays
        """
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
        
    def regress(data, x_data = np.ones(1), y_data = np.ones(1)):
        """
            Uses the given x_data and y_data arrays to create a linear regression plot.
        
            Parameters:
                x_data  :   1darray
                    The desired x_data array
                y_data  :   1darray
                    The desired y_data array
            
            Returns: plt.show()
                        Opens an external window that shows the linear regression plot of the given data. 
        """
        if np.array_equiv(testarray, y_data):
            y_data = np.full_like(x_data, 5)
        
        print("Given x data:", x_data)
        print("Given y data:", y_data)
        
        plt.title('pyplot best fit & linear regression plot')
        plt.plot(x_data, y_data, 'o')
        plt.plot(x_data, data.A + data.B * x_data)
        plt.show()
    
    def standdistgraph(data, x_data = np.ones(1)):
        """
            Uses the given x_data array to create a standard distribution graph.
            Currently only works for x data.
        
            Parameters:
                x_data  :   1darray
                    The desired x_data array
                
            Returns: plt.show()
                        Opens an external window that shows the standard distribution graph
        """
        
        gdata = x_data

        plt.hist(gdata, bins = len(gdata), density = True, alpha = 0.6, color = 'c', align = 'mid')
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, data.x_mean, data.sigma_x)
        plt.plot(x, p, 'k', linewidth = 2)
        title = "x data mean = %.2f, stddev x = %.2f, 2 stddev x = %.2f" % (data.x_mean, data.sigma_x, 2 * data.sigma_x)
        plt.title(title)
        plt.show()
