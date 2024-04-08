# API Reference

## *class* physerror.Data(user_x_data: ndarray, user_y_data: ndarray)

An initializer and container dataclass that is initialized and reused
by the user and other classes, as well as their methods. There are many
attributes that calculate common statistical errors, constants, and
general error propagation methods, and one class Method to find, document,
and delete any data points that exist outside the standard 2 \* sigma outlier
“limit”.

* **Parameters:**
  * **user_x_data** (*np.ndarray*) – []
  * **user_y_data** (*np.ndarray*) – []

### Attributes
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

#### NOTE
Will update this soon-ish

### Methods
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

## *class* physerror.Graphs

Allows the user to create various graphs from the user_data
pulled from Data. There is no \_\_init_\_ method for this Class.

### Methods
#### datahist(gtitle='Graph')

Uses the given dataframe built from x_data and y_data during
initalization to create one or two histograms. There is also
the option to turn the graphs into standard distribution
graphs (currently a WIP).

* **Parameters:**
  * **user_data** ([*Data*](#physerror.Data)) – Requires the user to pass in an instance of
    Data to make use of the user’s data.
  * **gtitle** (*str* *,* *optional*) – The desired graph title. Defaults to “Graph”.
* **Returns:**
  Opens an external window that displays the
  histogram(s).
* **Return type:**
  plt.show()

#### errbargraph(gtitle='Graph')

Uses the given dataframe built from x_data and y_data during
initalization to create an error bar plot, making use of
the sigma_x value as the constant error.

* **Parameters:**
  * **user_data** ([*Data*](#physerror.Data)) – Requires the user to pass in an instance of
    Data to make use of the user’s data.
  * **gtitle** (*str* *,* *optional*) – The desired graph title. Defaults to “Graph”.
* **Returns:**
  Opens an external window that displays the
  error bar graph.
* **Return type:**
  plt.show()

#### linreg(gtitle='Graph')

Uses the given x_data and y_data arrays to create a linear
regression plot.

* **Parameters:**
  * **user_data** ([*Data*](#physerror.Data)) – Requires the user to pass in an instance of
    Data to make use of the user’s data.
  * **gtitle** (*str* *,* *optional*) – The desired graph title. Defaults to “Graph”.
* **Returns:**
  Opens an external window that shows the linear
  regression plot of the given data.
* **Return type:**
  plt.show()

#### resid(gtitle='Graph')

Uses user_data._df to create a residuals scatter plot
via the seaborn sns.residplot method. The graph’s
title can optionally be customized.

* **Parameters:**
  * **user_data** ([*Data*](#physerror.Data)) – Requires the user to pass in an instance of
    Data to make use of the user’s data.
  * **gtitle** (*str* *,* *optional*) – The desired graph title. Defaults to “Graph”.
* **Returns:**
  Opens an external window that displays the residuals scatter plot.
* **Return type:**
  plt.show()

#### sctrplot(gtitle='Graph', marktype='D', markc='c', markedge='k')

Uses the given x_data and y_data to create a scatter plot
via matplot.pyplot’s scatter method. Customization options
are available, similar to the original pyplot method.

* **Parameters:**
  * **user_data** ([*Data*](#physerror.Data)) – Requires the user to pass in an instance of
    Data to make use of the user’s data.
  * **gtitle** (*str* *,* *optional*) – The desired graph title. Defaults to “Graph”.
  * **marktype** (*str* *,* *optional*) – The desired marker style. Defaults to “D”.
    See link for all available matplotlib markers:
    [https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers](https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers)
  * **markc** (*str* *,* *optional*) – The desired marker color. Defaults to ‘c’.
    See link for all matplotlob colors:
    [https://matplotlib.org/stable/gallery/color/named_colors.html](https://matplotlib.org/stable/gallery/color/named_colors.html)
  * **markedge** (*str* *,* *optional*) – The desired marker edge color. Defaults to ‘k’.
    See link for all matplotlob colors:
    [https://matplotlib.org/stable/gallery/color/named_colors.html](https://matplotlib.org/stable/gallery/color/named_colors.html)
* **Returns:**
  Opens an external window that displays the scatter plot.
* **Return type:**
  plt.show()

### physerror.csvreader()
