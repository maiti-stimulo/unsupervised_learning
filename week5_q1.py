# -*- coding: utf-8 -*-
import random
import numpy as np
from scipy.spatial.distance import cdist
import pylab
#generem una variable per simplificar escritura de codi
al_2d = np.atleast_2d
# al generar uns nombres aleatoris sempre obtindriem resusltats diferents, fents més complicat el càlcul.
# np.random.seed(1023), sempre ens donarà els mateixos nombres aleatoris
# aquests valors sempre estan entre 0 i 1
np.random.seed(1023)

#//////////////////////////////////////////////////////////////////////////////////
#Iniciem el K cluster amb centres x, utilitzan les dades aleatories
#//////////////////////////////////////////////////////////////////////////////////

def K_means(X, K):
# kmeans algorithm
# k = nombre de clusters
# X = centroides
    #np.random.choice:Generates a random sample from a given 1-D array
    #X.shape[0]:: agafem les dades de la primera columna de la matriu
    centroids = X[np.random.choice(X.shape[0], K, replace=False),:]
    convergence = False


    while not convergence:
        prev_centroids = centroids.copy()
        #Y serà el valor que representara a tot el cluster
        #Y = [nearest_centroid(point, centroids) for point in X]

	Y = []
	for point in X:
	    Y.append(nearest_centroid(point, centroids))
	    #append = anexar.
            #Append values to the end of an array
            #https://docs.scipy.org/doc/numpy/reference/generated/numpy.append.html
            #anexem les dades agrupades per clusters a la matriu Y

        centroids = recompute_centroids(Y, X)
        # quan els centroides s'estabilitzen es pq s'ha de complir que els centroides antics son = als nous
        # la funció recalcular centroides esta definida a la linia 77
        convergence = (prev_centroids == centroids).all()
        # A la línia 40 recalculem els centroides i a la 43 els comparem amb els que havíem guardat prèviament.Si són iguals, la variable "convergence" agafarà el valor "true" i ja no tornarem a entrar al bucle (línia 27). Ho fem aquí perquè és la última línia del bucle, on s'acostuma a recalcular les condicions d'entrada de nou al bucle
    return centroids
    # ens retorna els centroides

#//////////////////////////////////////////////////////////////////////////////////
#Assignem a acada punt els seu centroide més proper
#//////////////////////////////////////////////////////////////////////////////////

def nearest_centroid(feature, C):
    #distances = [cdist(al_2d(c_k), al_2d(feature)) for c_k in C]
    # generem una matriu per les distàncies
    distances = []
    # for c_k in C: vol dir tots els centroides que trobem dins de C
    for c_k in C:
        distances.append(cdist(al_2d(c_k), al_2d(feature)))
        #al generar la variable al_2d podem simplificar el codi, en cas de no haver-ho fet, ho hauriem d'escriure com la linia següent.
        #distances.append(cdist(np.atleast_2d(c_k), np.atleast_2d(feature)))
        #cdist és una funció que retorna la distància entre dos punts donats:
        #https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
        #atleast::View inputs as arrays with at least two dimensions.
        #https://docs.scipy.org/doc/numpy/reference/generated/numpy.atleast_2d.html
        #returns an array, or tuple of arrays, each with a.ndim >= 2. Copies are avoided where possible, and views with two or more dimensions are returned.
    #En resum, necessitem que dataset i sample tinguin el mateix número de columnes (és a dir, número de coordenades) per a que cdist funcioni, i atleast_2d ens fa això.

    return np.argmin(distances)
    #Returns the indices of the minimum values along an axis.
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmin.html

#//////////////////////////////////////////////////////////////////////////////////
#Recalculem cada centroide segons la mitjana dels punts assignats
#//////////////////////////////////////////////////////////////////////////////////

def recompute_centroids(Y, X):
    #Return a new array of given shape and type, filled with zeros.
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html
    # K= nombre de clusters
    # C ens genera una matriu d'aquest tipus [[0,0][0,0][0,0]
    # C_count ens genera una matriu  [0,0,0]
    #C_count és una variable "comptador" on guardem el número de punts assignats a cada centroide. Si K=3, llavors C_count guarda els punts assignats a Ck=0, Ck=1 i Ck=2.
    C = np.zeros((K,columns))
    C_count = np.zeros(K)
    #print 'C'
    #print C
    #print 'C_count'
    #print C_count

    #zip::Make an iterator that aggregates elements from each of the iterables.
    #https://docs.python.org/3/library/functions.html#zip
    #
    Z = zip(Y, X)
    # aquest for  ens serveix per contar els element que formarant part d'un clúster determinat.
    # en el cas que un element de la primera columna  sigui igual al K, afegueix l'element a aquell clúster.
    # C_count, és un contador d'elements de cada clúster
    for k in range(K):
        for z in Z:
            if z[0] == k:
                C[k] = C[k]+ z[1]
                C_count[k] = C_count[k] + 1

    C = np.divide(C, al_2d(C_count).T)
    # La T es l'operació de transposar matrius.
    # https://ca.wikipedia.org/wiki/Matriu_transposada
    # Divide arguments element-wise.
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.divide.html
    

    # ens retorna els 3 centroides finals més estables
    return C


if __name__ == '__main__':
    rows = 50
    columns = 2
    K = 3
    dataset = np.atleast_2d(np.random.random((rows,columns)))

    print 'Solució Q1:'
    # raw_imput ens guarda qualsevol informació dintre un string
    # per contiunar amb el pograma hem de prèmer return
    raw_input('prem return per veure les dades aleatories')
    # imprimim les dades que hem escollit aleatoriament amb el random (1023)
    print 'Dataset: %d files, %d columnes' % (rows, columns)
    print dataset
    print '\n'

    raw_input('prem return per veure els K-means trobats')
    C = K_means(dataset, K)
    print C
    print '\n'
