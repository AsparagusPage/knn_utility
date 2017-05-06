#!usr/bin/env python3

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def get_bestk(data_mat, num_ks):
    '''
    Returns the number of nearest neighbors that results in the least number of
    errors that occur in the 'leave one point out' test.
    data_mat is the matrix of training data points whose last column is the
    (numerical) labels of each point
    num_ks is the number of numbers of clusters to test (from 1 to num_ks
    clusters).
    '''
    errors = get_errors(data_mat, num_ks)

    return errors.index(min(errors))+1, errors

def get_errors(data_mat, num_ks):
    errors = [num_mistakes(data_mat, k+1) for k in range(num_ks)]

    return errors

def num_mistakes(data_mat, k):
    labeled_data = np.array(data_mat)
    last_column = labeled_data.shape[1] - 1
    data_points = labeled_data[:, :last_column]
    labels = labeled_data[:, last_column]
    count = 0
    for i in range(labeled_data.shape[0]-1):
        one_gone_data = np.delete(data_points, i, 0)
        one_gone_labels = np.delete(labels, i, 0)
        nbs = KNeighborsClassifier(n_neighbors = k)
        nbs.fit(one_gone_data, one_gone_labels)
        guess = nbs.predict(np.reshape(data_points[i,:],(-1,data_points.shape[1])))
        if guess != labels[i] :
            count+=1

    return count
