# oosPCA
Online Oversampling Principal Component Analysis Implementations

-----------------------------------------------------------------------------------------

Description:

This method of data mining, based upon the statistical procedure principal component analysis (PCA), 
is useful for identifying values in data which difer signifcantly from all other members in the dataset. 
While the standard PCA algorithm works to reduce dimensions in a dataset by singling out the data items
with the most variance, oosPCA works by oversampling each data instance and determining if it changes 
the overall variance of the dataset by a great amount. If it does, it must be an outlier. Additionally, 
oosPCA was designed to process data very quickly, making it useful for large datasets and giving 
it an advantage over many other anomaly detection data mining algorithms.

-----------------------------------------------------------------------------------------

Requirements:

This script is designed to run on the Bluenose server using Python. All necessary 
libraries are installed on Bluenose.

If you are running it from another machine, you may need to install:
Python 2.7
Pandas (https://pandas.pydata.org/)
Numpy (http://www.numpy.org/)
sklearn
itertools

Data files:
Ensure all data files are in a text format (.txt, .csv, etc)
Store data files in the data subdirectory
Items should be arranged in columns with should be delimited with commas without headers, 
and with the target/classification data in the last column
Headers should be stored in a separate .names file with the header name on each line 
proceeded by its type ("continuous." for numbers or "symbolic" for every else), like this:

protocol_type: symbolic.
duration: continuous.

Do not include the classification header in the .names file.
You may put the possible classifications in the first line of the .names file, but do not 
include colons in this line. See the kddcup.names file for an example

-----------------------------------------------------------------------------------------

Usage Instructions:

- Open a terminal in the folder containing the osPCA.py files
- Enter "python osPCA.py" in the command line
- Prompts will request the following information:
        - The name of the training file
        - The name of the data file you wish to test (see above for data format)
        - The header .names file
        	- if your headers are included, the program will ask you to specify the target
        	  classification and will make its best guess at which columns are numerical
        - The normal classification
        - If symbolic data is included in your dataset, you may chose to break up your
          data by one of these symbolic attributes
        - The percentage of your data to use for training
- The accuracy result will print to the terminal. The full results will print to csv files,
depending on how many categories of data you chose to work with named "results_[CATEGORY].csv"

-----------------------------------------------------------------------------------------

How it works:

The program is run from osPCA.py

Loops will promp for the data filenames and the target attribute.
The dataframe and names files you specify will be read using Pandas.


names_to_col.make_col(path)
	- Uses the .names file to organize the headers and dataframe
	
The data is split into training and testing data and outliers are dropped from the 
training set
	
categorize_data(df, continuous_headers, symbolic_headers, categorize_by):
	- If you specify a category to split data, this function will break up the dataframe
	  in the training data and call the pca_vector and outlier_point functions necessary 
	  for each category
	- returns the split dataframes, their dominant eigenvectors, and the outlier scores
	  for each point
	
un_categorize_data(df, symbolic_headers):
	- If no categories are specified, this function performs the same work to the training
	  data as categorize_data, but does not split up the dataframe
	
pca_vector(df, target):
	- runs the first steps of the PCA algorithm
	- normalized and calculates the covariance matrix of the dataframe
	- calculates the eigenvalues and eigenvectors of the dataframe and sorts them
	- returns the dominant eigenvector (associated with largest eigenvalue)

calculate_outlier_score(a, b):
	- returns the variance between two eigenvectors 
	
outlier_point(row, df, eigvec, target):
	- for training data
	- This function oversamples the testing point (row) by 10% of the dataframe size
	- Gets the resulting eigenvector and tests against the dataset eigenvector using 
	calculate_outlier_score
	- returns just the outlier score
	
calculate_thredshold(array):
	- using the outlier scores from the training data, this function returns the mean and 
	standard deviation which are used later for testing data
	
check_point(row, df, eigvec, target, normal_mean, normal_sd):
	- oversamples the testing data point by 10% the size of the testing data
	- Gets the resulting eigenvector and tests against the dataset eigenvector using 
	calculate_outlier_score, then uses check_threshold to get outlier classification 

After the testing data has finished, accuracies are printed and full dataset results are 
written to CSV

-----------------------------------------------------------------------------------------

Limitations:

- This program does not implement the online portion of the algorithm
- This program requires a very specific style of data formatting to ensure that only 
numerical values are testing
- Best accuracy results require a very large set of training data
