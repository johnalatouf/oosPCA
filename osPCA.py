# this script takes a dataframe of data, computes the dominant eigenvector,
# then to test it, duplicates some of the points in the dataframe, computes a second
# eigenvector and compares the angle between them
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from imblearn.ensemble import EasyEnsemble
import names_to_col
from itertools import repeat

# reads the file specified by path and returns dataframe
def read_data_file(path):
    fullpath = "./%s" % path
    csv = pd.read_csv(fullpath, delim_whitespace=True, nrows=1000)
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
    # mean_vec = np.mean(X_std, axis=0)

    # get the covariance matrix
    cov_mat = np.cov(X_std.T)



    # get the eigenvec and eigenval
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    # singlular vector decomposition
    # u, s, v = np.linalg.svd(X_std.T)

    # make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    # print eig_pairs

    # sort the (eigenvalue, eigenvector) tuples from high to low
    # TODO - sort gets a weird error with KDD data so find the largest val the manual way
    # eig_pairs.sort()
    # eig_pairs.reverse()

    # lets just get the biggest pair this way
    eig_val_highest = eig_pairs[0][0]
    eig_dominant = eig_pairs[0][1]
    for e in eig_pairs:
        if e[0] > eig_val_highest:
            eig_val_highest = e[0]
            eig_dominant = e[1]

    # tot = sum(eig_vals)
    # var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
    # cum_var_exp = np.cumsum(var_exp)

    # choose the top two eiganvectors to go from 4d to 2d
    # matrix_w = np.hstack((eig_pairs[0][1].reshape(4, 1), eig_pairs[1][1].reshape(4, 1)))

    # return eig_pairs[0][1]
    return eig_dominant

# borrowed from here: https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
def unit_vector(vector):
    return vector / np.linalg.norm(vector)


# returns the radian difference between angles
def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
# calculate the score of outlierness by given two vector
def calculate_outlier_score(a, b):
	"""Takes 2 vectors a, b and returns the score of outlierness
	"""
	dot_product = np.dot(a, b)
	norm_a = np.linalg.norm(a)
	norm_b = np.linalg.norm(b)
	return 1 - abs(dot_product / (norm_a * norm_b))

#return  
def pca_algorithm(df, target):
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
    
    #threshold
    varRetained = 0.95
    #return eig_pairs[0][1]
    k = len(var_i[var_i < (varRetained * 100)])
    #check threshold 
    if self.calculate_outlier_score(eigvec, eigvec2) < varRetained :
        # compute the reduced dimensional features by projction        
        U_reduced = u[:, : k]
        Z = np.transpose(U_reduced) * X

    return Z, U_reduced

#random over-sampling function
def random_oversampling(df, target):
    # drop the target from the dataframe
    df_data = df.drop(target, axis=1)

    x = df_data.values
    y = df[target].values
    
    # creates an ensemble of data set by randomly under-sampling the original set
    ee = EasyEnsemble(random_state=0, n_subsets=10)
    X_resampled, y_resampled = ee.fit_sample(x, y)
    return X_resampled, y_resampled



# pick out the groups to test based on symbolic data
# can probably do this for each symbolic category
def categorize_data(df, continuous_headers, symbolic_headers, categorize_by):
    # for each in the category
    dataframes = {}
    eig_vecs = {}
    for category in  df[categorize_by].unique():
        df_cat = df[df[categorize_by] == category]              # only use this value in category
        # TODO - I believe we can keep binary symbolic headers, but I'm not sure how to handle them yet
        df_cat.drop(symbolic_headers, axis=1, inplace=True)     # drop symbolic headers
        dataframes[category] = df_cat
        eig_vecs[category] = pca_vector(df_cat, target)
    return dataframes, eig_vecs


# duplicate point by 10%, add to the cluster, check the eigenvalue
# pandas iterrows is terrible and this can probably be done better using apply()
def check_points(df, eigvec, target):
    scores = []
    df_to_test = df.copy()
    ten_percent = int(len(df)*0.1)
    print ten_percent
    for idx, point in df.iterrows():

        # make a list of 10% copies of this point and concat it to the test data
        pointlist = list(repeat(point.tolist(), ten_percent))
        df_point = pd.DataFrame(data=pointlist, columns=df.columns)
        df_to_test = pd.concat([df_to_test, df_point])

        new_eigvec = pca_vector(df_to_test, target)
        outlier_score = calculate_outlier_score(eigvec, new_eigvec)
        scores.append([outlier_score, point[target]])
        # print "score: %s, class: %s" % (outlier_score, point[target])
        df_to_test = df.copy()
    sc = pd.DataFrame(scores)
    print len(sc)
    return sc

# duplicate point by 10%, add to the cluster, check the eigenvalue
# pandas iterrows is terrible and this can probably be done better using apply()
def check_point(row, df, eigvec, target):

    scores = []
    df_to_test = df.copy()
    ten_percent = int(len(df)*0.1)


    pointlist = list(repeat(row.tolist(), ten_percent))
    df_point = pd.DataFrame(data=pointlist, columns=df.columns)
    df_to_test = pd.concat([df_to_test, df_point])

    new_eigvec = pca_vector(df_to_test, target)
    # TODO - this is the real one
    outlier_score = calculate_outlier_score(eigvec, new_eigvec)
    # outlier_score = angle_between(eigvec, new_eigvec)

    scores.append([outlier_score, row[target]])
    # print "score: %s, class: %s" % (outlier_score, point[target])
    # TODO - there's more to the outlier score
    print outlier_score
    return outlier_score


def testingrow(row, df):
    print row
    print row["protocol_type"]
    print len(df)


if __name__ == "__main__":
    # path = sys.argv[1]
    # target = sys.argv[2]
    #
    # # read the dataframe
    # df = read_data_file(path)

    # read the total lines, but only work with the first ~2000 normal points

    # df = pd.read_csv(filepath_or_buffer='data/kddcup.data_10_percent',
    #                  header=None,
    #                  sep=',', nrows=50000)

    df = pd.read_csv(filepath_or_buffer='data/kddcup5000.csv',
                     header=None,
                     sep=',')
    #
    # df.to_csv("kddcup50000.csv")

    print len(df)

    # read in the first 5000 lines
    # df = pd.read_csv(filepath_or_buffer='data/kddcup5000.csv',
    #                  header=None,
    #                  sep=',')


    #add the headers so it is easier to work with
    df.columns, continuous, symbolic = names_to_col.make_col("data/kddcup.names")
    df.dropna(how="all", inplace=True)  # drops the empty line at file-end

    train_size = int(len(df) * 0.1)
    df_train = df.drop(df.index[[0, train_size]])
    df_test = df.drop(df.index[-train_size])
    # print df_test
    # print df_test


    # in this instance, there are no headers so grab the last column as class
    target = "class"
    target_normal = "normal."
    categorizedby = "protocol_type"



    # drop outliers from the training data
    df_train = df_train[df_train[target] == target_normal]


    # categorize by protocol type and get the clusters and their eigvectors
    clusters, eigvecs = categorize_data(df_train, continuous, symbolic, categorizedby)

    outlier_scores = pd.DataFrame()

    # TODO - do this differently

    # for each row in testing data, compute an outlier
    # test against same protocol
    print eigvecs.keys()
    # print df_test[categorizedby].unique()
    # print df_test
    # df_test.apply(testingrow, args = (df_train,), axis=1)
    # TODO - this is the good part

    # categorize the test data
    df_tests = {}
    for cat in df_test[categorizedby].unique():
        print cat
        df_tests[cat] = df_test[df_test[categorizedby] == cat]
        df_tests[cat].drop(symbolic, axis=1, inplace=True)

        df_tests[cat]['outlier_score'] = df_tests[cat].apply((lambda x: check_point(x,clusters[cat], eigvecs[cat], target)), axis=1)


        # print out some results to a csv
        # print df_tests[cat]
        csvname = "kdd_outlier_scores_%s" % cat
        df_tests[cat].to_csv(csvname)

        #testing the results
        df_outlier = df_tests[cat][df_tests[cat][target] != "normal."]
        df_normal = df_tests[cat][df_tests[cat][target] == "normal."]
        print len(df_normal)
        print len (df_outlier)
        print "%s normal difference in dot product mean: %s" % (cat, (df_normal["outlier_score"].mean()))
        print "%s outlier difference in dot product mean: %s" % (cat, (df_outlier["outlier_score"].mean()))

    # for key,val in clusters.items():
    #     outlier_scores = pd.concat([outlier_scores, check_points(val, eigvecs[key], target)])

    # a way to print these and read them
    # outlier_scores.to_csv("output.csv")
    #
    # outlier_scores_normal = outlier_scores[outlier_scores[1] == "normal."]
    # outlier_scores_outlier = outlier_scores[outlier_scores[1] != "normal."]
    #
    # print "normal %s" % outlier_scores_normal[0].mean()
    # print "outliers %s" % outlier_scores_outlier[0].mean()

    # eigvec2 = pca_vector(df_extra, target)
    # print eigvec2
    #
    # angle =  angle_between(eigvec, eigvec2)
    # print angle
    # print np.rad2deg(angle)
    
    
        
