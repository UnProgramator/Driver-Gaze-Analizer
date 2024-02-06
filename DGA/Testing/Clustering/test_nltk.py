from nltk.cluster import KMeansClusterer, euclidean_distance, cosine_distance
import numpy
import math

vectors = [numpy.array(f) for f in [[2, 1], [1, 3], [4, 7], [6, 7]]]
means = [[4, 3], [5, 5]]

clusterer1 = KMeansClusterer(2, cosine_distance, initial_means=means)
clusters1 = clusterer1.cluster(vectors, True, trace=True)

print("Clustered:", vectors)
print("As:", clusters1)
print("Means:", clusterer1.means())
print()

vectors = [numpy.array(f) for f in [[3, 3], [1, 2], [4, 2], [4, 0], [2, 3], [3, 1]]]

# test k-means using the euclidean distance metric, 2 means and repeat
# clustering 10 times with random seeds

clusterer2 = KMeansClusterer(3, euclidean_distance, repeats=10)
clusters2 = clusterer2.cluster(vectors, True)
print("Clustered:", vectors)
print("As:", clusters2)
print("Means:", clusterer2.means())
print()

vectors = [numpy.array(f) for f in [[3, 3], [1, 2], [4, 2], [4, 0], [2, 3], [3, 1]]]

# test k-means using the euclidean distance metric, 2 means and repeat
# clustering 10 times with random seeds

clusterer2 = KMeansClusterer(3, euclidean_distance, repeats=10)
clusters2 = clusterer2.cluster(vectors, True)
print("Clustered:", vectors)
print("As:", clusters2)
print("Means:", clusterer2.means())
print()


vectors = [numpy.array(f) for f in [[3, 3], [1, 2], [4, 2], [4, 0], [2, 3], [3, 1]]]

def my_distance(u, v):
    diff = v-u
    return math.sqrt(numpy.dot(diff, diff))


clusterer3 = KMeansClusterer(3, my_distance, repeats=10)
clusters3 = clusterer3.cluster(vectors, True)
print("Clustered:", vectors)
print("As:", clusters3)
print("Means:", clusterer3.means())
print()


# classify a new vector
vector = numpy.array([3, 3])

print("classify(%s):" % vector, end=" ")
print(clusterer1.classify(vector))
print()

print("classify(%s):" % vector, end=" ")
print(clusterer2.classify(vector))
print()

print("classify(%s):" % vector, end=" ")
print(clusterer3.classify(vector))
print()


##############################################################################################################################

vectors = [numpy.array(f) for f in [['a', 'a'], ['a', 'b'], ['a', 'e'], ['b', 'b']]]

def my_distance(u, v):
    
    u = numpy.array(list(map(ord, u)))
    v = numpy.array(list(map(ord, v)))

    diff = v-u
    return math.sqrt(numpy.dot(diff, diff))


clusterer4 = KMeansClusterer(2, my_distance, repeats=10)
clusters4 = clusterer4.cluster(vectors, True)
print("Clustered:", vectors)
print("As:", clusters4)
print("Means:", clusterer4.means())
print()