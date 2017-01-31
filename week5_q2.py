# -*- coding: utf-8 -*-
#/////////////////////////////////////////////////////////////////////////////////////
#Fins linia 44 l'exercici funciona igual que el Q1
#////////////////////////////////////////////////////////////////////////////////////
import random
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

al_2d = np.atleast_2d

def K_means(X, K):
    centroids = X[np.random.choice(X.shape[0], K, replace=False),:]
    convergence = False

    while not convergence:
        prev_centroids = centroids.copy()
        Y = [nearest_centroid(point, centroids) for point in X]
        centroids = recompute_centroids(Y, X)
        convergence = (prev_centroids == centroids).all()

    return centroids, Y

def nearest_centroid(feature, C):
    distances = [cdist(al_2d(c_k), al_2d(feature)) for c_k in C]
    return np.argmin(distances)

def recompute_centroids(Y, X):
    C = np.zeros((K,X.shape[1]))
    C_count = np.zeros(K)

    Z = zip(Y, X)

    for k in range(K):
        for z in Z:
            if z[0] == k:
                C[k] = C[k]+ z[1]
                C_count[k] = C_count[k] + 1

    C = np.divide(C, al_2d(C_count).T)

    return C

if __name__ == '__main__':
    # These options determine the way floating point numbers, arrays and other NumPy objects are displayed.
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.set_printoptions.html
    #formatter : dict of callables
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

    # preparem iris.data tal i com el necessitem per fer els càlculs
    iris_data = np.array([l.strip('\n\r').split(',') for l in open('iris.data')])
    iris_data_no_labels = np.array([map(float, l.split(',')[:-1]) for l in open('iris.data')])

    labels = iris_data[:,-1]

    # Mantenim K=3, que és el que ens demanen
    K = 3

    print 'Solució Q2 amb les dues primeres dimensions del data set:'
    raw_input('Visualitzem el dataset, la seva forma es(%d rows, %d columns), prem retorn per visualitzar-lo' % (iris_data_no_labels.shape[0], iris_data_no_labels.shape[1]))
    print 'Dataset:'
    print iris_data
    print '\n'

    raw_input('prem return per veure els K-means trobats')
    # indiquem que utilitzarem les dues primeres dimensions de la iris data
    C, Y_c = K_means(iris_data_no_labels[:,0:2], K)

    for i in range(K):
        print 'Centroides %d = %s' % (i, C[i])

    print '\n'
    raw_input('Trobem MSE per cada clúster,prem retorn per visualitzar-los')
    Z = zip(iris_data_no_labels[:,0:2], Y_c)
    distances = []

    for z in Z:
        point = z[0]
        cluster = z[1]
        centroid = C[cluster]
        distance = cdist(al_2d(point), al_2d(centroid))
        distances.append((distance,cluster))

    MSE = np.zeros(K)
    for i in range(K):
        x = [(d[0]-0)**2 for d in distances if d[1] == i]
        sum_x = np.sum(x)
        MSE[i] = sum_x/len(x)
        print 'MSE[%d] = %4.4f' % (i,MSE[i])

    print '\n'
    raw_input('Comparativa amb els centres dels clústers')
    Centers = []
    for i in range(K):
        max_0 = -100.0
        max_1 = -100.0
        min_0 = 100.0
        min_1 = 100.0
        x = [z[0] for z in Z if z[1] == i]
        for point in x:
            if point[0] > max_0:
                max_0 = point[0]
            if point[1] > max_1:
                max_1 = point[1]
            if point[0] < min_0:
                min_0 = point[0]
            if point[1] < min_1:
                min_1 = point[1]
        center_point = [(max_0 + min_0)/2.0, (max_1 + min_1)/2.0]
        Centers.append(center_point)

    Centers = np.array(Centers)

    Z = zip(C,Centers)

    distance_to_center = []
    for z in Z:
        centroid = z[0]
        center = z[1]
        distance = cdist(al_2d(centroid), al_2d(center))
        distance_to_center.append(distance)

    for i in range(K):
        print 'Distància del centre[%d] al centroide[%d] = %4.4f' % (i,i,distance_to_center[i])

    print '\n'

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////
    print 'Solució Q2 amb totes les dimensions del data set:'

    raw_input('prem return per veure els K-means trobats amb totes les dimensions del data set')
    # indiquem que utilitzarem totes dimensions de les dades que no son etiquetes
    C, Y_c = K_means(iris_data_no_labels[:,:], K)

    for i in range(K):
        print 'Centroides tot %d = %s' % (i, C[i])

    print '\n'
    raw_input('Trobem MSE per cada clúster,prem retorn per visualitzar-los')
    Z = zip(iris_data_no_labels[:,:], Y_c)
    distances = []

    for z in Z:
        point = z[0]
        cluster = z[1]
        centroid = C[cluster]
        distance = cdist(al_2d(point), al_2d(centroid))
        distances.append((distance,cluster))

    MSE = np.zeros(K)
    for i in range(K):
        x = [(d[0]-0)**2 for d in distances if d[1] == i]
        sum_x = np.sum(x)
        MSE[i] = sum_x/len(x)
        print 'MSE tot[%d] = %4.4f' % (i,MSE[i])

    print '\n'
    raw_input('Comparativa amb els centres dels clústers')
    Centers = []
    for i in range(K):
        max_0 = -100.0
        max_1 = -100.0
        min_0 = 100.0
        min_1 = 100.0
        x = [z[0] for z in Z if z[1] == i]
        for point in x:
            if point[0] > max_0:
                max_0 = point[0]
            if point[1] > max_1:
                max_1 = point[1]
            if point[0] < min_0:
                min_0 = point[0]
            if point[1] < min_1:
                min_1 = point[1]
        center_point = [(max_0 + min_0)/2.0, (max_1 + min_1)/2.0]
        Centers.append(center_point)

    Centers = np.array(Centers)

    Z = zip(C,Centers)

    distance_to_center = []
    for z in Z:
        centroid = z[0]
        center = z[1]
        distance = cdist(al_2d(centroid), al_2d(center))
        distance_to_center.append(distance)

    for i in range(K):
        print 'Distància del centre[%d] al centroide[%d] = %4.4f' % (i,i,distance_to_center[i])

    print '\n'


    raw_input('imprimeix les dades')
    print 'imprimint (utilitzarem tres columnes per a visualitzar-ho en 3d)'

    # gràfics 3d
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # punts a utilitzar
    xt = iris_data_no_labels[:, 0]
    yt = iris_data_no_labels[:, 1]
    zt = iris_data_no_labels[:, 1]
    ax.scatter(xt, yt, zt, s=10, c='r')

    # els centroides
    ax.scatter(C[:,0], C[:,1], C[:,1], s=80, c='y')

    # llegendes
    ax.set_xlabel('Sepal long en cm')
    ax.set_ylabel('Sepal amplada en cm')
    ax.set_zlabel('Petal long en cm')

    # imprimir-ho
    plt.show()

