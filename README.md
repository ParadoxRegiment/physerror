# physerror

physerror is an educational/support module written by Alexander Ritzie for the use of physics students.
It was originally created for the BPHYS 231 Intro to Experimental Physics class at University of Washington
Bothell.

If you are new to coding, look below under [Examples](#examples) and [Usage](#usage) for how to load
your data, pull calculated values, and generate graphs.

## Documentation
To view all the documentation here as well as a full API reference, please visit the [Documentation Pages](docs/_build/markdown/getstart.md)

## Installation

Download the __physerror.py__ file and save it in the same directory as your working directory. Use
`import physerror as phyerr` to import the module into your file.

## Examples
```python
import numpy as np
import physerror as phyerr

# Loading data
x_1 = np.arange(5)
y_1 = np.arange(5)
exp_data = phyerr.Data(x_1, y_1)

# Finding and printing outliers
x_out, y_out = exp_data.outlier()
print(x_out, "\n", y_out)

# 

# Generating a linear regression graph
phyerr.Graphs.linreg(exp_data)
### Once it's done running, the method automatically
### outputs a plotted graph

# Generating a histogram plot
phyerr.Graphs.datahist(exp_data)
### Once it's done running, the method automatically
### outputs a plotted graph
```

## Usage

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

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
