# this script takes a dataframe of data, computes the dominant eigenvector,
# then to test it, duplicates some of the points in the dataframe, computes a second
# eigenvector and compares the angle between them
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
import names_to_col
from itertools import repeat
import time

pd.options.mode.chained_assignment = None  # clears up a false-positive warning with pandas

# # reads the file specified by path and returns dataframe
# def read_data_file(path):
#     fullpath = "./%s" % path
#     csv = pd.read_csv(fullpath, nrows=1000)
#     return csv

# do the pca algorithm to get the dominant vector
# borrowed from the tutorial
def pca_vector(df, target):

    # drop the target from the dataframe
    df_data = df.drop(target, axis=1)

    x = df_data.values
    if target in df:
        y = df[target].values
    X_std = StandardScaler().fit_transform(x)

    # get the covariance matrix
    cov_mat = np.cov(X_std.T)

    # get the eigenvec and eigenval
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    # make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

    # sort the (eigenvalue, eigenvector) tuples from high to low
    # lets just get the biggest pair this way
    eig_val_highest = eig_pairs[0][0]
    eig_dominant = eig_pairs[0][1]
    for e in eig_pairs:
        if e[0] > eig_val_highest:
            eig_val_highest = e[0]
            eig_dominant = e[1]

    return eig_dominant


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

    return mean, sd


# check the point against the threshold
def check_threshold(mean, sd, point):
    outlier = None
    if (point < mean - 2 * sd) or (point > mean + 2 * sd):
        outlier = True
    return outlier

# pick out the groups to test based on symbolic data
# can probably do this for each symbolic category
def categorize_data(df, continuous_headers, symbolic_headers, categorize_by):
    # for each in the category
    dataframes = {}
    eig_vecs = {}
    outlier_scores = {}
    for category in df[categorize_by].unique():
        df_cat = df[df[categorize_by] == category]              # only use this value in category
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


# in the case of uncategorized data, we just want to do one big clump
def un_categorize_data(df, symbolic_headers):
    # for each in the category
    dataframes = {}
    eig_vecs = {}
    outlier_scores = {}
    df_cat = df.drop(symbolic_headers, axis=1)     # drop symbolic headers
    dataframes["all"] = df_cat
    eig_vecs["all"] = pca_vector(df_cat, target)

    # we need to get mean outlier scores for all normal training data
    # to use for computing mean and sd for threshold testing
    df_cat['outlier_score'] = df_cat.apply(
        (lambda x: outlier_point(x, df_cat, eig_vecs["all"], target)), axis=1)
    outlier_scores["all"] = df_cat['outlier_score'].tolist()
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
    outlier_score = calculate_outlier_score(eigvec, new_eigvec)

    scores.append([outlier_score, row[target]])
    outlier_classification = check_threshold(normal_mean, normal_sd, outlier_score)
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
    outlier_score = calculate_outlier_score(eigvec, new_eigvec)

    scores.append([outlier_score, row[target]])
    return outlier_score


  ####################################
 ###    TEST WITH ONLINE        #####
###################################

#The following methods are for running the algorithm online. 
#I believe they exploit the similarity of the PCA algorithm with
#the computation of least squares to simplify and speed up computation 
#of new eigenvectors calculated from oversampling. 
#These functions, if ever tied into main program, should go somewhere between
#the calculation of the primary eigenvector and result output. That is, I think the 
#flow should be:
# [Get primary eigenvector with train data] -> [calculate global mean/s.d.]
# -> [find global values xproj and y using startonline] -> 
# [feed in test points one at a time to runonline, and use function to compute new eigenvector in-place]
# -> [compare new eigenvector with standard] -> [get results]


def startonline(domev, gmean, dat):
    #compute the xbar_proj and y needed for runonline function
    #domev: dominant eigenvectors
    #gmean: global mean

    xproj = 0
    y = 0

    for index, row in dat.iterrows():
        yj = transpose(domev)*(row[t] - gmean)
        xproj = yj*(dat[t] - gmean)

        y += yj**2

    return xproj, y

def runonline(xproj, y):
    #Input:
        #U: matrix of k dominant eigenvectors [principal dir]
        #x_i: data points
        #y: 
    #xbar_i = x_i - mean
    #xbar_p: target data instance? 
    #beta = 1/(n*r) [is weighting factor]
        #

    return


def testingrow(row, df):
    print row
    print row["protocol_type"]
    print len(df)

# find possible binary columns in the data
# returns a list of possible target attributes
def potentialNormals(df, attrib):
    targets = df[attrib].unique().tolist()
    return targets

if __name__ == "__main__":

    # imput params:
    datapath  = 'data/kddcup50000.csv'      # the data csv
    namefile = 'data/kddcup.names'          # the headers for the csv w/ continuous vs symbolic labeling
    target = "class"                        # the target attribute
    target_normal = "normal."               # the *normal* attribute
    categorizedby = "protocol_type"         # the symbolic attribute to categorize by, if desired
    categorized_data = False                # whether or not to split the data
    continuous = []                         # holds continuous headers (numbers)
    symbolic = []                           # holds symbolic headers (strings, objects, bools, etc)
    clusters = {}                           # holds dataframe clusters
    eigvecs = {}                            # holds eigvecs by category
    outliers = {}                           # holds outliers by category
    training_size_perc = 0.1                # what percentage of the data is for training?



    # make a nice printout and ask for the data
    print ("\n-----------------------------------------------------")
    print ("\n\tCSCI 4144 Group Project: Online Oversampling PCA")
    print ("\tJohna Latouf, Chaoran Zhou, Seth Piercey")
    print ("\n\tType 'quit' to exit.")
    print ("\n-----------------------------------------------------")
    while True:
        # path: the data file name entered
        datapath = raw_input("\nPlease enter a data file (eg: data/kddcup.data_10_percent): ")
        if datapath == "quit":
            sys.exit(0)
        try:
            open(datapath, 'r')
            break;
        except IOError:
            print("\nCan't open the file.")
            continue;
        except OSError:
            print("\nCan't open the file.")
            continue;

    while True:
        # path: the name file name entered
        namefile = raw_input("\nPlease enter a .names file (eg: data/kddcup.names) or (\'i\' to ignore): ")
        if namefile == "quit":
            sys.exit(0)
        if namefile == "i":
            break;
        try:
            open(namefile, 'r')
            break;
        except IOError:
            print("\nCan't open the file.")
            continue;
        except OSError:
            print("\nCan't open the file.")
            continue;


    # account for included headers
    if namefile != 'i':
        # read the data file
        df = pd.read_csv(filepath_or_buffer=datapath,
                         header=None,
                         sep=',')
        # add the headers so it is easier to work with
        try:
            df.columns, continuous, symbolic = names_to_col.make_col(namefile)
        except ValueError:
            print "These files do not match"
            sys.exit(0)
        df.dropna(how="all", inplace=True)  # drops the empty line at file-end
    else:
        # read the data file w/ headers
        df = pd.read_csv(filepath_or_buffer=datapath, sep=',')
        df_num = df.select_dtypes(include=[np.number])
        df_string = df.select_dtypes(exclude=[np.number])
        continuous = df_num.columns.tolist()
        symbolic = df_string.columns.tolist()

        # asking for target
        count = 1
        for t in symbolic:
            print "\t%s. %s" % (count, t)
            count += 1
        target_int = -1
        while True:
            target_int = raw_input(
                "Please select your target classification by number (eg: 1): ")
            if target_int == "quit":
                sys.exit(0)
            if target_int.isdigit() and int(target_int) <= len(symbolic) and int(target_int) > 0:
                categorized_data = True
                break
            else:
                print "You must select a valid digit"
                continue
        if categorized_data:
            target = symbolic[int(target_int) - 1]
            symbolic.pop(int(target_int) - 1)


    # ask for the "normal" or inlier value from thet arget
    vals = potentialNormals(df, target)
    count = 1
    for t in vals:
        print "\t%s. %s" % (count, t)
        count += 1
    target_int = -1
    while True:
        target_int = raw_input("Please select the normal classification by number (eg. 1): ")
        if target_int == "quit":
            sys.exit(0)
        if target_int.isdigit() and int(target_int) <= len(vals) and int(target_int) > 0:
            break
        else:
            print "You must select a valid digit"
            continue
    target_normal = vals[int(target_int)-1]

    # now if we want to further break up this data by a category, we can do it like this:
    if len(symbolic) > 0:
        count = 1
        for t in symbolic:
            print "\t%s. %s" % (count, t)
            count += 1
        target_int = -1
        while True:
            target_int = raw_input(
                "Your data has non-numerical attributes that will be discarded. If you would like to split your data by one of these attributes, please choose a number (eg. 1), otherwise type \'i\' to ignore: ")
            if target_int == "quit":
                sys.exit(0)
            if target_int == "i":
                break
            if target_int.isdigit() and int(target_int) <= len(symbolic) and int(target_int) > 0:
                categorized_data = True
                break
            else:
                print "You must select a valid digit"
                continue
        if categorized_data:
            categorizedby = symbolic[int(target_int) - 1]


    # what size is the training?
    while True:
        # the min support entered
        training_size_perc = raw_input("\nPlease the percentage of this set to use as training data in the form of a decimal (0 to 1): (eg: .1) ")
        if training_size_perc == "quit":
            sys.exit(0)
        try:
            training_size_perc = float(training_size_perc)
            if training_size_perc < 0 or training_size_perc > 1:
                print("\nA valid input is between 0.00-1.00 (eg: .1).")
                continue;
            break;
        except ValueError:
            print("\nA valid input is a float between 0.00-1.00 (eg: .1).")
            continue;

    # get the training size, we want about 10%
    train_size = int(len(df) * float(training_size_perc))
    # print train_size
    df_train = df[:train_size]
    # print len(df_train)
    df_test = df[:-train_size]
    # print len(df_test)



    # drop outliers from the training data
    df_train = df_train[df_train[target] == target_normal]

    print "size of normal training data: %s" % len(df_train)

    # depending on the input, categorize or don't categorize
    if categorized_data:
        clusters, eigvecs, outliers = categorize_data(df_train, continuous, symbolic, categorizedby)
    # categorize by protocol type and get the clusters and their eigvectors and a list of outlier scores
    else:
        clusters, eigvecs, outliers = un_categorize_data(df_train, symbolic)

    outlier_scores = pd.DataFrame()

    # TODO - this is the good part

    # categorize the test data
    df_tests = {}
    if categorized_data:
        for cat in df_test[categorizedby].unique():
            print cat
            df_tests[cat] = df_test[df_test[categorizedby] == cat]
            df_tests[cat].drop(symbolic, axis=1, inplace=True)

            # calculate the normal mean and sd
            if cat in outliers:
                normal_mean, normal_sd = calculate_thredshold(outliers[cat])
            else:
                continue

            # print "%s %s %s %s %s %s" % (cat, len(clusters[cat]), eigvecs[cat], target, normal_mean, normal_sd)

            df_tests[cat]['outlier_score'], df_tests[cat]['outlier_class'] = zip(*df_tests[cat].apply(
                (lambda x: check_point(x, clusters[cat], eigvecs[cat], target, normal_mean, normal_sd)), axis=1))


            # print out some results to a csv
            # print df_tests[cat]
            csvname = "results_%s.csv" % cat
            df_tests[cat].to_csv(csvname)

            #testing the results
            df_outlier = df_tests[cat][df_tests[cat][target] != target_normal]
            df_normal = df_tests[cat][df_tests[cat][target] == target_normal]
    else:
        cat = "all"
        df_tests[cat] = df_test.copy()
        df_tests[cat].drop(symbolic, axis=1, inplace=True)

        # calculate the normal mean and sd
        normal_mean, normal_sd = calculate_thredshold(outliers[cat])


        # print "%s %s %s %s %s %s" % (cat, len(clusters[cat]), eigvecs[cat], target, normal_mean, normal_sd)

        df_tests[cat]['outlier_score'], df_tests[cat]['outlier_class'] = zip(*df_tests[cat].apply(
            (lambda x: check_point(x, clusters[cat], eigvecs[cat], target, normal_mean, normal_sd)), axis=1))

        # print out some results to a csv
        # print df_tests[cat]
        csvname = "results_%s.csv" % cat
        df_tests[cat].to_csv(csvname)

        # testing the results
        df_outlier = df_tests[cat][df_tests[cat][target] != target_normal]
        df_normal = df_tests[cat][df_tests[cat][target] == target_normal]

    # now find accuracy
    if categorized_data:
        for cat in df_test[categorizedby].unique():
            df_correct_outlier = df_tests[cat][(df_tests[cat][target] != target_normal) & (df_tests[cat]["outlier_class"] == True)]
            df_missed_normal = df_tests[cat][(df_tests[cat][target] == target_normal) & (df_tests[cat]["outlier_class"] == True)]
            df_missed_outlier = df_tests[cat][(df_tests[cat][target] != target_normal) & (df_tests[cat]["outlier_class"] != True)]
            df_correct_normal = df_tests[cat][(df_tests[cat][target] == target_normal) & (df_tests[cat]["outlier_class"] != True)]

            num_outliers = float(len(df_tests[cat][df_tests[cat][target] != target_normal][target].tolist()))
            num_normals = float(len(df_tests[cat][df_tests[cat][target] == target_normal][target].tolist()))

            if num_outliers > 0:
                print "%s correct id'd outliers: %s" % (cat, float(len(df_correct_outlier)) / num_outliers)
            if num_normals > 0:
                print "%s false positive outliers: %s" % (cat, float(len(df_missed_normal)) / num_normals)
            if num_outliers > 0:
                print "%s missed outliers: %s" % (cat, float(len(df_missed_outlier)) / num_outliers)
            if num_normals > 0:
                print "%s correct id'd normals: %s" % (cat, float(len(df_correct_normal)) / num_normals)
    else:
        cat = "all"
        df_correct_outlier = df_tests[cat][
            (df_tests[cat][target] != target_normal) & (df_tests[cat]["outlier_class"] == True)]
        df_missed_normal = df_tests[cat][
            (df_tests[cat][target] == target_normal) & (df_tests[cat]["outlier_class"] == True)]
        df_missed_outlier = df_tests[cat][
            (df_tests[cat][target] != target_normal) & (df_tests[cat]["outlier_class"] != True)]
        df_correct_normal = df_tests[cat][
            (df_tests[cat][target] == target_normal) & (df_tests[cat]["outlier_class"] != True)]

        num_outliers = float(len(df_tests[cat][df_tests[cat][target] != target_normal][target].tolist()))
        num_normals = float(len(df_tests[cat][df_tests[cat][target] == target_normal][target].tolist()))

        if num_outliers > 0:
            print "%s correct id'd outliers: %s" % (cat, float(len(df_correct_outlier)) / num_outliers)
        if num_normals > 0:
            print "%s false positive outliers: %s" % (cat, float(len(df_missed_normal)) / num_normals)
        if num_outliers > 0:
            print "%s missed outliers: %s" % (cat, float(len(df_missed_outlier)) / num_outliers)
        if num_normals > 0:
            print "%s correct id'd normals: %s" % (cat, float(len(df_correct_normal)) / num_normals)

    
    
        
