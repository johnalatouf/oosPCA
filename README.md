# oosPCA
Online Oversampling Principal Component Analysis Implementations

This method of data mining, based upon the statistical procedure principal component analysis (PCA), 
is useful for identifying values in data which difer signifcantly from all other members in the dataset. 
While the standard PCA algorithm works to reduce dimensions in a dataset by singling out the data items
with the most variance, oosPCA works by oversampling each data instance and determining if it changes 
the overall variance of the dataset by a great amount. If it does, it must be an outlier. Additionally, 
oosPCA was designed to process data very quickly, making it useful for large datasets and giving 
it an advantage over many other anomaly detection data mining algorithms.
