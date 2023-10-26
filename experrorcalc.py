import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

x_data = np.array([])
y_data = np.array([])

class Calculations:
    """
    To access values and functions, must do [variable] = Calculuations(ndarray, ndarray)
    To access functions, do [above variable].[function name]
    
    Available values:
        N: size of x data
        delta: the calculated delta constant
        A: the calculated A constant
        B: the calculated B constant
        x_mean: the calculated x data mean
        y_mean: the calculated y data mean
        sigma_x: the calculated sigma of x data
        sigma_y: the calculated sigma of y data
        sigma_A: the calculated sigma of constant A
        sigma_B: the calculated sigma of constant B
    """
    
    def __init__(self, x_data, y_data):
        self.N = len(x_data)
        self.delta = self.N * sum(x_data ** 2) - (sum(x_data)) ** 2
        self.A = ((sum(x_data ** 2) * sum(y_data)) - (sum(x_data) * sum(x_data * y_data))) / self.delta
        self.B = (self.N * sum(x_data * y_data) - (sum(x_data) * sum(y_data))) / self.delta
        self.x_mean = abs(np.mean(x_data))
        self.y_mean = abs(np.mean(y_data))
        self.sigma_y = np.sqrt((1/(self.N - 2)) * sum((y_data - self.A - (self.B * x_data)) ** 2))
        self.sigma_A = self.sigma_y * np.sqrt(sum(x_data ** 2) / self.delta)
        self.sigma_B = self.sigma_y * np.sqrt(self.N / self.delta)
        self.sigma_x = np.sqrt(sum((x_data - self.x_mean) ** 2) / (self.N - 1))
    
    def outlier(data, x_data, y_data):
        """
            Checks for and prints out any outliers in included arrays within the 2 * sigma limit.
            
            Parameters:
                x_data  :   ndarray
                    The desired x_data array
                y_data  :   ndarray
                    The desired y_data array
            
            Returns: print()
                        Prints out x_outliers and y_outliers in the form of ndarrays
        """
        
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
            x_outliers = 'None'
        
        if np.size(y_outliers) == 0:
            y_outliers = 'None'

        print(x_outliers)
        print(y_outliers)
        
    def regress(data, x_data, y_data):
        plt.title('pyplot best fit & linear regression plot')
        plt.plot(x_data, y_data, 'o')
        plt.plot(x_data, data.A + data.B * x_data)
        plt.show()
    
    def standdist(data, x_data):
        gdata = x_data

        plt.hist(gdata, bins = len(gdata), density = True, alpha = 00.6, color = 'c', align = 'mid')
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, data.x_mean, data.sigma_x)
        plt.plot(x, p, 'k', linewidth = 2)
        title = "x data mean = %.2f, stddev x = %.2f, 2 stddev x = %.2f" % (data.x_mean, data.sigma_x, 2 * data.sigma_x)
        plt.title(title)
        plt.show()