# this script takes a dataframe of data, computes the dominant eigenvector,
# then to test it, duplicates some of the points in the dataframe, computes a second
# eigenvector and compares the angle between them
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler

# reads the file specified by path and returns dataframe
def read_data_file(path):
    fullpath = "./%s" % path
    csv = pd.read_csv(fullpath, delim_whitespace=True)
    return csv

# do the pca algorithm to get the dominant vector
# borrowed from the tutorial
def pca_vector(df, target):
    # drop the target from the dataframe
    df_data = df.drop(target, axis=1)

    x = df_data.values
    y = df[target].values
    X_std = StandardScaler().fit_transform(x)

    # get the mean vector
    mean_vec = np.mean(X_std, axis=0)

    # get the covariance matrix
    cov_mat = np.cov(X_std.T)

    # get the eigenvec and eigenval
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    # singlular vector decomposition
    u, s, v = np.linalg.svd(X_std.T)

    # make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

    # sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort()
    eig_pairs.reverse()

    tot = sum(eig_vals)
    var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)

    # choose the top two eiganvectors to go from 4d to 2d
    matrix_w = np.hstack((eig_pairs[0][1].reshape(4, 1), eig_pairs[1][1].reshape(4, 1)))

    return eig_pairs[0][1]

# borrowed from here: https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
def unit_vector(vector):
    return vector / np.linalg.norm(vector)

# returns the radian difference between angles
def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

if __name__ == "__main__":
    # path = sys.argv[1]
    # target = sys.argv[2]
    #
    # # read the dataframe
    # df = read_data_file(path)

    df = pd.read_csv(filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                     header=None,
                     sep=',')

    df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
    df.dropna(how="all", inplace=True)  # drops the empty line at file-end

    target="class"

    eigvec =  pca_vector(df, target)
    print eigvec
    df_more = df[df["class"] == "Iris-setosa"]
    df_extra = pd.concat([df, df_more], ignore_index=True)

    eigvec2 = pca_vector(df_extra, target)
    print eigvec2

    angle =  angle_between(eigvec, eigvec2)
    print angle
    print np.rad2deg(angle)
