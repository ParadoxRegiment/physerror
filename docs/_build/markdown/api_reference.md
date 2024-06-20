# API Reference

### *class* physerror.Data(user_x_data: ~numpy._typing._array_like._SupportsArray[~numpy.dtype[~typing.Any]] | ~numpy._typing._nested_sequence._NestedSequence[~numpy._typing._array_like._SupportsArray[~numpy.dtype[~typing.Any]]] | bool | int | float | complex | str | bytes | ~numpy._typing._nested_sequence._NestedSequence[bool | int | float | complex | str | bytes] = <factory>, user_y_data: ~numpy._typing._array_like._SupportsArray[~numpy.dtype[~typing.Any]] | ~numpy._typing._nested_sequence._NestedSequence[~numpy._typing._array_like._SupportsArray[~numpy.dtype[~typing.Any]]] | bool | int | float | complex | str | bytes | ~numpy._typing._nested_sequence._NestedSequence[bool | int | float | complex | str | bytes] = <factory>, data_type: str = <factory>)

An initializer and container dataclass that is initialized and reused
by the user and other classes, as well as their methods. There are many
attributes that calculate common statistical errors, constants, and
general error propagation methods, and two class methods.

#### delta

[]

* **Type:**
  float

#### A

[]

* **Type:**
  float

#### B

[]

* **Type:**
  float

#### x_mean

[]

* **Type:**
  float

#### x_best

[]

* **Type:**
  float

#### y_mean

[]

* **Type:**
  float

#### y_best

[]

* **Type:**
  float

#### sigma_y

[]

* **Type:**
  float

#### sigma_A

[]

* **Type:**
  float

#### sigma_B

[]

* **Type:**
  float

#### sigma_x

[]

* **Type:**
  float

#### sigma_x_best

[]

* **Type:**
  float

#### sigma_y_best

[]

* **Type:**
  float

#### sigma_x_mean

[]

* **Type:**
  float

#### sigma_y_mean

[]

* **Type:**
  float

#### sigma_frac

[]

* **Type:**
  float

* **Parameters:**
  * **user_x_data** (*ArrayLike*) – []
  * **user_y_data** (*ArrayLike*) – []

#### NOTE
Will update this soon-ish

#### export()

Exports error analysis calculations to either an Excel workbook
or JSON file based on user’s choice.

#### outlier()

A method that creates two empty arrays then searches the
cls.x_data and cls.y_data arrays for values that are outside
the standard 2 \* sigma outlier “limit”.

* **Returns:**
  * *np.ndarray* – An array that contains either the outliers that were found
    in the user’s x_data, or a string stating no outliers were
    found.
  * *np.ndarray* – An array that contains either the outliers that were found
    in the user’s y_data, or a string stating no outliers were
    found.

### *class* physerror.FileReaders

Container class for methods that read in and parse data files.

#### csvreader()

Reads in a csv file selected via a tkinter file explorer window.
Assumes there is no index column. Data should be organized into columns
rather than rows.

* **Returns:**
  * *ndarray* – A NumPy array containing the data in the leftmost column of the
    passed in data file.
  * *ndarray* – A NumPy array containing the data in the second column of the
    passed in data file.

#### excelreader()

Reads in an Excel Workbook selected via a tkinter file explorer window.
Assumes there is no index column. Data should be organized into columns
rather than rows.

* **Returns:**
  * *ndarray* – A NumPy array containing the data in the leftmost column of the
    passed in data file.
  * *ndarray* – A NumPy array containing the data in the second column of the
    passed in data file.

### *class* physerror.Graphs

Allows the user to create various graphs from the user_data
pulled from Data.

#### graph_title

String used as the title for any graphing function that is run.
Defaults to “Graph”.

* **Type:**
  str = “Graph”

#### title_size

Numerical value used as the title font size for any graphing
function that is run. Defaults to 11.

* **Type:**
  int = 11 | float

#### x_label

String used as the x-axis label for any graphing function with
points that is run. Defaults to “x label”

* **Type:**
  str = “x label”

#### y_label

String used as the y-axis label for any graphing function with
points that is run. Defaults to “y label”.

* **Type:**
  str = “y label”

#### p_color

Either a single string, list, or ArrayLike of strings that is used
to color the points in any graphing function with points that is run.
Defaults to “cyan”.

* **Type:**
  str = “cyan” | list[str] | ArrayLike[str]

#### line_color

A string used to color the line in the linreg and resid functions.
Defaults to “black”.

* **Type:**
  str = “black”

#### errbar_color

A string used to color the error bars in the errbargraph function.
Defaults to “red”.

* **Type:**
  str = “red”

#### dist_check

A string used to determine if the datahist function will generate
normal distrbution histograms. Only accepts “Yes” and “No”, and
defaults to “No”.

* **Type:**
  str = “Yes” | “No”

#### dataset_check

An integer used to determine how many datasets will be generated
from the datahist function. Only accepts 1 and 2, and defaults to 1.

* **Type:**
  int = 1 | 2

#### hist_color

Either a single string, list, or ArrayLike of strings to determine
the color of the the histogram(s) from the datahist function. Defaults
to “green”.

* **Type:**
  str = “green” | list[str] | ArrayLike[str]

#### dbl_pend_line

A string used to change the color of specifically the double pendulum
function’s line color. Defaults to “lime”.

* **Type:**
  str = “lime”

#### dbl_pend_trace

A string used to change the color of specifically the double pendulum
function’s trace line color. Defaults to “black”.

* **Type:**
  str = “black”

#### datahist(user_data: [Data](#physerror.Data))

Generates a histogram of one or two sets of data pulled
from the Data class using pandas’ DataFrame.hist method.

* **Parameters:**
  **user_data** ([*Data*](#physerror.Data)) – Requires the user to pass in an instance of
  Data to make use of the user’s data.

#### dbl_pend(theta_0: float, phi_0: float, theta_dot_0=0, phi_dot_0=0, anim_type=0)

Generates either a point mass or bar mass double pendulum
animation based on the pass in initial values. Angles are read
as the angle between the bar/string and an imaginary horizontal
line going through the point.

Point mass calculations and animation code were taken from
matplotlib’s documentation:
[https://matplotlib.org/stable/gallery/animation/double_pendulum.html](https://matplotlib.org/stable/gallery/animation/double_pendulum.html)

* **Parameters:**
  * **theta_0** (*float*) – Initial angle of the top bar/string.
  * **phi_0** (*float*) – Initial angle of the bottom bar/string.
  * **theta_dot_0** (*int = 0* *(**optional* *)*) – Initial velocity of the top bar/string. Defaults to 0.
  * **phi_dot_0** (*int = 0* *(**optional* *)*) – Initial velocity of the bottom bar/string. Defaults to 0.
  * **anim_type** (*int = 0* *|* *1* *(**optional* *)*) – Optional variable that determines the type of double
    pendulum that will be used. Defaults to 0 for Point Mass,
    accepts 1 for Bar Mass.

#### errbargraph(user_data: [Data](#physerror.Data))

Uses the x data and y data from Data to create a point-based
with error bars on each point. Error size is the sigma_x value
calculated in Data.

* **Parameters:**
  **user_data** ([*Data*](#physerror.Data)) – Requires the user to pass in an instance of
  Data to make use of the user’s data.

#### linreg(user_data: [Data](#physerror.Data))

Uses the given x_data and y_data arrays to create a linear
regression plot.

* **Parameters:**
  **user_data** ([*Data*](#physerror.Data)) – Requires the user to pass in an instance of
  Data to make use of the user’s data.

#### resid(user_data: [Data](#physerror.Data))

Uses user_data._df to create a residuals scatter plot
via the seaborn sns.residplot method. The graph’s title
can optionally be customized.

* **Parameters:**
  **user_data** ([*Data*](#physerror.Data)) – Requires the user to pass in an instance of
  Data to make use of the user’s data.

#### sctrplot(user_data: [Data](#physerror.Data))

Uses the given x_data and y_data to create a scatter plot
via matplot.pyplot’s scatter method. Customization options
are available, similar to the original pyplot method.

* **Parameters:**
  **user_data** ([*Data*](#physerror.Data)) – Requires the user to pass in an instance of
  Data to make use of the user’s data.
