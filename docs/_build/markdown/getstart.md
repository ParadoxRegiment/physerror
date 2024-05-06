<!-- physerror documentation master file, created by
sphinx-quickstart on Tue Apr  2 12:58:06 2024.
You can adapt this file completely to your liking, but it should at least
contain the root `toctree` directive. -->

# Getting Started

## Installation

To use physerror as a module, download the `physerror.py` file and save it in the same directory as your working directory. Use
`import physerror as phyerr` to import the module into your file. Otherwise, download the file and save it in any given directory.

## Examples

These examples apply only if you wish to import physerror into another project and use it there. If you only want to use the module
to do error calculations and create graphs, it is highly recommended that you simply run the file itself for its built-in user menu.

```python
import numpy as np
import physerror as phyerr

# Loading data
x_1 = np.arange(5)
y_1 = np.arange(5)
exp_data = phyerr.Data(x_1, y_1)
### Note that x_1 and y_1 are *not* required. The module will ask
### what type of file you would like to import, and if you select
### "Manual" it will use whatever has been passed into the variable

# Finding and printing outliers
x_out, y_out = exp_data.outlier()
print(x_out, "\n", y_out)

# Exporting error analysis calculations to an Excel or JSON file
exp_data.export()

# Generating a linear regression graph
phyerr.Graphs.linreg(exp_data)
### Once it's done running, the method automatically
### outputs a plotted graph

# Generating a histogram plot
phyerr.Graphs.datahist(exp_data)
### Once it's done running, the method automatically
### outputs a plotted graph
```

When using physerror as a self contained script, the file will start up a “command-line-interface” menu. This allows
users to easily choose what type of file they would like to import (this method only allows importing), modify predetermined
properties for each function, and run multiple functions back-to-back.

Below is a series of images that show an example of importing a CSV file, exporting the error data to an Excel file, and running
the Graphs.linreg() method. The example screenshots are, for the most part, applicable to any other given function as well.

**Selecting a file type**

![image](docs/docs_screenshots/file_select.png)

**Reading in a CSV file**

![image](docs/docs_screenshots/csv_read.png)

**Exporting error data to an Excel file**

![image](docs/docs_screenshots/excel_export.png)

**Property editing menu - Graphs.linreg() example**

![image](docs/docs_screenshots/prop_edit_linreg.png)

**Editing specific values - Title example**

![image](docs/docs_screenshots/prop_edit_gen.png)

**Graphs.linreg() output after being run**

![image](docs/docs_screenshots/linreg_example.png)

## Usage

physerror can be used either as a Python module or as a self contained Python script. To see how it should
be used as a module, see the code-block below. To use it as a script, simply run the file either in a Python
editor (i.e. Visual Studio Code, Spyder) or in a command line (i.e. Command Prompt window, PowerShell). Example
images can be found above in Examples.

```python
import physerror as phyerr

# Initialization class that calculates standard error information
# and must be passed into any Graphs method to work
phyerr.Data()
# This method asks the user if they would like to read in a .csv file
# If the user answers yes, a File Explorer window will open so you can
# directly select the file rather than type in the path. Please be sure
# to know where your .csv file is located if you wish to use it.

# Returns outliers in the passed-in x data and y data
phyerr.Data.outlier()

# Generates a linear regression graph using the data passed into
# phyer.Data
phyerr.Graphs.linreg()

# Generates a standard error bar point graph using the data passed
# into phyer.Data
phyerr.Graphs.errbargraph()

# Generates a histogram graph using the data passed into phyer.Data
# The user can choose whether or not the method graphs only x_data
# or both x_data and y_data, and if the histograms should have a
# standard distrbution line
phyerr.Graphs.datahist()

# Generates a scatter plot using the data passed into phyer.Data
phyerr.Graphs.sctrplot()

# Generates a residuals scatter plot using the data passed into
# phyer.Data
phyerr.Graphs.resid()

# An extra method used to read a csv file into two arrays. Though
# not necessarily useful for data or error analysis, it may be useful
# to some students who would like to easily split their .csv files into
# arrays.
phyerr.csvreader()
```
