# this script takes a dataframe of data, computes the dominant eigenvector,
# then to test it, duplicates some of the points in the dataframe, computes a second
# eigenvector and compares the angle between them
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
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

# calculate the mean and standdev of an array 
#and detect a points(its projected coefficient onto this eigenvector) with this threshold
def calculate_thredshold(array):
    """
    Takes an array of the normal data instances projected onto the dominant eigenvector
    reject the outlier points by eliminating any points that were above (Mean + 2*SD) 
    and any points below (Mean - 2*SD) before plotting the frequencies
    return boolean outlier as true or false 
    """
    outlier = None
    elements = np.array(array)

    mean = np.mean(elements, axis=0)
    sd = np.std(elements, axis=0)

    # TODO - I'm going to move this so that a mean can be determined ahead of time and
    # we just have to calculate it once, then check the point in another function later
    # if (point < mean - 2 * sd) or (point > mean + 2 * sd) :
    #     outlier = True
    # #print (outlier)
    # return outlier
    return mean, sd


# check the point against the threshold
def check_threshold(mean, sd, point):
    outlier = None
    if (point < mean - 2 * sd) or (point > mean + 2 * sd):
        outlier = True
    # print (outlier)
    return outlier

# pick out the groups to test based on symbolic data
# can probably do this for each symbolic category
def categorize_data(df, continuous_headers, symbolic_headers, categorize_by):
    # for each in the category
    dataframes = {}
    eig_vecs = {}
    outlier_scores = {}
    for category in  df[categorize_by].unique():
        df_cat = df[df[categorize_by] == category]              # only use this value in category
        # TODO - I believe we can keep binary symbolic headers, but I'm not sure how to handle them yet
        df_cat.drop(symbolic_headers, axis=1, inplace=True)     # drop symbolic headers
        dataframes[category] = df_cat
        eig_vecs[category] = pca_vector(df_cat, target)

        # we need to get mean outlier scores for all normal training data
        # to use for computing mean and sd for threshold testing
        df_cat['outlier_score'] = df_cat.apply(
            (lambda x: outlier_point(x, df_cat, eig_vecs[category], target)), axis=1)
        outlier_scores[category] = df_cat['outlier_score'].tolist()
        df_cat.drop('outlier_score', axis=1, inplace=True)
    return dataframes, eig_vecs, outlier_scores


# duplicate point by 10%, add to the cluster, check the eigenvalue outlier score
# test the outlier score against the normal mean and sd scores to detect outlier
# returns outlier score and outlier true/false
def check_point(row, df, eigvec, target, normal_mean, normal_sd):
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
    outlier_classification = check_threshold(normal_mean, normal_sd, outlier_score)
    print "score: %s, classification: %s" % (outlier_score, outlier_classification)
    return outlier_score, outlier_classification

# duplicate point by 10%, add to the cluster, check the eigenvalue for outlier score
def outlier_point(row, df, eigvec, target):

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

    #   TODO
    ## UNCOMMENT BEFORE PASSING IN ##
    #print "Data files are assumed to be in \'./data/\' folder"
    #fname = raw_input("Name of data file: ")
    #datapath = 'data/'+fname
    datapath = 'data/kddcup5000.csv'
    df = pd.read_csv(filepath_or_buffer=datapath,
                     header=None,
                     sep=',', nrows=50000)
    #
    # df.to_csv("kddcup50000.csv")



    # read in the first 5000 lines
    # df = pd.read_csv(filepath_or_buffer='data/kddcup5000.csv',
    #                  header=None,
    #                  sep=',')


    #TODO uncomment before passing in
    #namefile = raw_input("What is the data column name file (\'i\' to ignore): ")
    namefile = 'data/kddcup.names'
    if namefile != 'i':
        #add the headers so it is easier to work with
        df.columns, continuous, symbolic = names_to_col.make_col(namefile)
        df.dropna(how="all", inplace=True)  # drops the empty line at file-end

    train_size = int(len(df) * 0.1)
    print train_size
    df_train = df[:train_size]
    print len(df_train)
    df_test = df[:-train_size]
    print len(df_test)


    # in this instance, there are no headers so grab the last column as class
    target = "class"
    target_normal = "normal."
    categorizedby = "protocol_type"


    # drop outliers from the training data
    df_train = df_train[df_train[target] == target_normal]


    # categorize by protocol type and get the clusters and their eigvectors and a list of outlier scores
    clusters, eigvecs, outliers = categorize_data(df_train, continuous, symbolic, categorizedby)

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

        # calculate the normal mean and sd
        if cat in outliers:
            normal_mean, normal_sd = calculate_thredshold(outliers[cat])
        else:
            continue

        print "%s %s %s %s %s %s" % (cat, len(clusters[cat]), eigvecs[cat], target, normal_mean, normal_sd)

        df_tests[cat]['outlier_score'], df_tests[cat]['outlier_class'] = zip(*df_tests[cat].apply(
            (lambda x: check_point(x, clusters[cat], eigvecs[cat], target, normal_mean, normal_sd)), axis=1))


        # print out some results to a csv
        # print df_tests[cat]
        csvname = "kdd_outlier_scores_%s.csv" % cat
        df_tests[cat].to_csv(csvname)

        #testing the results
        df_outlier = df_tests[cat][df_tests[cat][target] != "normal."]
        df_normal = df_tests[cat][df_tests[cat][target] == "normal."]
        print len(df_normal)
        print len (df_outlier)
        print "%s normal difference in dot product mean: %s" % (cat, (df_normal["outlier_score"].mean()))
        print "%s outlier difference in dot product mean: %s" % (cat, (df_outlier["outlier_score"].mean()))

    # now find accuracy
    for cat in df_test[categorizedby].unique():
        df_correct_outlier = df_tests[cat][(df_tests[cat][target] != target_normal) & (df_tests[cat]["outlier_class"] == True)]
        df_missed_normal = df_tests[cat][(df_tests[cat][target] == target_normal) & (df_tests[cat]["outlier_class"] == True)]
        df_missed_outlier = df_tests[cat][(df_tests[cat][target] != target_normal) & (df_tests[cat]["outlier_class"] != True)]
        df_correct_normal = df_tests[cat][(df_tests[cat][target] == target_normal) & (df_tests[cat]["outlier_class"] != True)]

        num_outliers = float(len(df_tests[cat][df_tests[cat][target] != target_normal][target].tolist()))
        num_normals = float(len(df_tests[cat][df_tests[cat][target] == target_normal][target].tolist()))


        print "%s correct id'd outliers: %s" % (cat, float(len(df_correct_outlier))/num_outliers)
        print "%s false positive outliers: %s" % (cat, float(len(df_missed_normal))/num_normals)
        print "%s missed outliers: %s" % (cat, float(len(df_missed_outlier))/num_outliers)
        print "%s correct id'd normals: %s" % (cat, float(len(df_correct_normal))/num_normals)

    
    
        
