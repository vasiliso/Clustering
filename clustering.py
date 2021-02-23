import pandas as pd
import csv
import random
import math
import sys


def point_dist(a, b):
    # returns the euclidean distanc between a and b
    squares = 0
    for i in range(1, (len(a) - 1)):
        squares = squares + (float(a[i]) - float(b[i]))**2
    return math.sqrt(squares)


def sorted_dist(data):
    # returns a list of all edges sorted by smallest distance
    # used by single_linkage
    distances = list()

    for i in range(0, len(data)):
        for j in range(i+1, len(data)):
            dist = point_dist(data[i], data[j])
            distances.append([i, j, dist])
    distances.sort(key=lambda x: x[2])
    return distances


def sq_dist(data):
    # returns a n*n matrix of the distances between all n data points
    distances = list()

    for i in range(0, len(data)):
        row = list()
        for j in range(0, len(data)):
            dist = point_dist(data[i], data[j])
            row.append(dist)
        distances.append(row)
    return distances


def single_linkage(filename, k):
    # single_linkage repeatedly takes the clustering which contain the two closest points and merges them, until k clustering remain
    initial_points, clustering = open_file(filename)
    distances = sorted_dist(initial_points)
    while (len(clustering) > k):
        # find the two closest clustering using the sorted distance list
        for i in range(0, len(clustering)):
            for point in clustering[i]:
                if point[0] == distances[0][0]:
                    cluster1 = i
                if point[0] == distances[0][1]:
                    cluster2 = i
        del(distances[0])
        if (cluster1 == cluster2) or (cluster1 == -1):
            continue
        # merge the two clustering
        for point in clustering[cluster2]:
            clustering[cluster1].append(point)
        del(clustering[cluster2])
    print_clustering(clustering)
    truth = actual_clustering(initial_points)
    hamming_distance = hamming(initial_points,
                               clustering, truth)
    print("Hamming distance from truth:", hamming_distance, "\n")
    return hamming_distance


def average_linkage(filename, k):
    # average_linkage repeatedly merges the two clustering with
    # the smallest average distance until k clustering remain
    initial_points, clustering = open_file(filename)
    distances = sq_dist(initial_points)
    while (len(clustering) > k):
        min_avg = 100000000000
        cluster1, cluster2 = -1, -1
        for i in range(0, len(clustering)):
            for m in range(i+1, len(clustering)):
                sum = 0
                for j in range(0, len(clustering[i])):
                    for l in range(0, len(clustering[m])):
                        sum = sum + distances[clustering[i]
                                              [j][0]][clustering[m][l][0]]
                avg = sum / (len(clustering[i])*len(clustering[m]))
                if avg < min_avg:
                    min_avg = avg
                    cluster1 = i
                    cluster2 = m
        if (cluster1 == cluster2) or (cluster1 == -1):
            continue
            # merging the clustering
        for point in clustering[cluster2]:
            clustering[cluster1].append(point)
        del(clustering[cluster2])
    print_clustering(clustering)
    truth = actual_clustering(initial_points)
    hamming_distance = hamming(initial_points,
                               clustering, truth)
    print("Hamming distance from truth:", hamming_distance, "\n")
    return hamming_distance


def lloyds_centroids(points, k):
    # this function selects the initial centroids for lloyd's method.
    #  It essentially just picks k random points
    random.seed()

    centroids = list()
    while len(centroids) < k:
        point = random.choice(points)
        new_centroid = point[:]
        new_centroid[0] = len(centroids)
        centroids.append(point)
    centroids.sort
    return centroids


def naive_kmeanspp_centroids(points, k):
    # this function selects the intial centroids for naive Kmeans++,
    # which assumes that the next centroid to be selected is the one
    # with the largest distance to its closest centroid.
    # It picks a random point then keeps finding the point with the maximum
    # distance from its closest centroid, and adding it to the centroids list
    centroids = list()
    random.seed()
    point = random.choice(points)
    new_centroid = point[:]
    new_centroid[0] = len(centroids)
    centroids.append(new_centroid)
    while len(centroids) < k:
        max_dist = 0
        for point in points:
            min_p2c_dist = 10000000
            for centroid in centroids:
                p2c_dist = point_dist(centroid, point)
                if p2c_dist < min_p2c_dist:
                    min_p2c_dist = p2c_dist
            if min_p2c_dist > max_dist:
                max_dist = min_p2c_dist
                max_pt = point
        new_centroid = max_pt[:]
        new_centroid[0] = len(centroids)
        centroids.append(new_centroid)
    return centroids


def kmeanspp_centroids(points, k):
    # this function selects the intial centroids for Kmeans++.
    # It picks a random point then keeps finding the next points with probability D(x)^2/(Sum(D(x)^2) for all x),
    # where D(x) is the distance between the point and its nearest centroid
    centroids = list()
    random.seed()
    point = random.choice(points)
    new_centroid = point[:]
    new_centroid[0] = len(centroids)
    centroids.append(new_centroid)
    while len(centroids) < k:
        distances = list()
        max_dist = 0
        for point in points:
            min_p2c_dist = 10000000
            for centroid in centroids:
                p2c_dist = point_dist(centroid, point)
                if p2c_dist < min_p2c_dist:
                    min_p2c_dist = p2c_dist
            distances.append(min_p2c_dist**2)
        dist_sum = sum(distances)
        for i in range(0, len(distances)):
            distances[i] = distances[i] / dist_sum
            if i > 1:
                distances[i] = distances[i] + distances[i - 1]
        rand = random.random()
        for i in range(0, len(distances)):
            if rand < distances[i]:
                new_centroid = points[i][:]
                new_centroid[0] = len(centroids)
                centroids.append(new_centroid)
                break
    return centroids


def find_centroids(initial_points, clustering):
    # this function finds the centroids from a given clustering
    centroids = list()
    for cluster in clustering:
        centroid = list()
        total = [0] * len(initial_points[0])
        if(len(cluster) != 0):
            for point in cluster:
                for j in range(1, len(point) - 1):
                    total[j] = total[j] + float(point[j])
            for i in range(0, len(total)):
                centroid.append(total[i] / len(cluster))
        else:
            for thing in total:
                thing = 0
                centroid.append(thing)
        centroid[0] = len(centroids)
        centroids.append(centroid)
    centroids.sort()
    return centroids


def kmeans(initial_points, k, method):
    # this function is the main kmeans algorithm
    # it gets an initial list of centroids depending on the method required, then
    # assigns each point in the data to a cluster according to its closest centroid
    # then it finds the centroids of these new clusters and repeats the process until
    # the centroids of two consecutive runs are the same
    if method == 'lloyds':
        centroids = lloyds_centroids(initial_points, k)
    if method == 'kmeanspp':
        centroids = kmeanspp_centroids(initial_points, k)
    if method == 'naivekmeanspp':
        centroids = naive_kmeanspp_centroids(initial_points, k)

    while True:
        clustering = list()
        for i in range(0, k):
            cluster = list()
            clustering.append(cluster)
        for point in initial_points:
            min_p2c_dist = 100000000000
            min_centr = -1
            for j in range(0, len(centroids)):
                dist = point_dist(point, centroids[j])
                if dist < min_p2c_dist:
                    min_p2c_dist = dist
                    min_centr = j
            clustering[min_centr].append(point)
        new_centroids = find_centroids(initial_points, clustering)
        if(new_centroids == centroids):
            break
        else:
            centroids = new_centroids
    return clustering, centroids


def run_kmeans(filename, k, method, runs):
    # this function runs the kmans function 100 times and selects the clustering with the lowest kmeans cost
    initial_points, clustering = open_file(filename)
    min_cost = 100000000000
    for i in range(runs):
        clustering, centroids,  = kmeans(initial_points, k, method)
        cost = kmeans_cost(clustering, centroids)
        if cost < min_cost:
            min_cost = cost
            best_clustering = clustering
    print_clustering(best_clustering)
    truth = actual_clustering(initial_points)
    hamming_distance = hamming(initial_points,
                               clustering, truth)
    print("Hamming distance from truth:", hamming_distance, "\n")
    return hamming_distance


def open_file(filename):
    # open file and ensure format of dadaset is correct
    delim = ','
    with open(filename + ".data", newline='') as f:
        reader = csv.reader(f, delimiter=delim)
        initial_points = list(reader)
    if len(initial_points[len(initial_points) - 1]) == 0:
        del(initial_points[len(initial_points) - 1])
    clustering = list()
    for i in range(0, len(initial_points)):
        point = initial_points[i]
        if filename == 'balance-scale':
            point.append(point[0])
        if filename == 'glass' or filename == 'balance-scale' or filename == 'yeast':
            del point[0]
        point.insert(0, i)
        new = list()
        new.append(point)
        clustering.append(new)
    return initial_points, clustering


def kmeans_cost(clustering, centroids):
    # return the sum of the square distances between each point and its corresponding centroid
    cost = 0
    for i in range(len(clustering)):
        for point in clustering[i]:
            cost = cost + point_dist(point, centroids[i])**2
    return cost


def hamming(initial_points, c1, c2):
    # loops through every edge and checks whether it is in-cluster or out of cluster in each clustering,
    # and returns the total number of mismatches
    h1 = 0
    h2 = 0
    for i in range(0, len(initial_points)):
        for j in range(i+1, len(initial_points)):
            cluster1 = -1
            cluster2 = -1
            cluster3 = -1
            cluster4 = -1
            for k in range(0, len(c1)):
                for point in c1[k]:
                    if point[0] == initial_points[i][0]:
                        cluster1 = k
                    if point[0] == initial_points[j][0]:
                        cluster2 = k
                    if(cluster1 != -1 and cluster2 != -1):
                        break
            for k in range(0, len(c2)):
                for point in c2[k]:
                    if point[0] == initial_points[i][0]:
                        cluster3 = k
                    if point[0] == initial_points[j][0]:
                        cluster4 = k
                    if(cluster3 != -1 and cluster4 != -1):
                        break
            if(cluster1 == cluster2 and cluster3 != cluster4):
                h1 = h1 + 1
            if(cluster1 != cluster2 and cluster3 == cluster4):
                h2 = h2 + 1
    return (h1+h2) / math.comb(len(initial_points), 2)


def print_clustering(clustering):
    print("new clustering")
    i = 0
    for cluster in clustering:
        cluster.sort()
        # print(len(cluster), i)
        i = i + 1
        # print(i, len(cluster))
        for point in cluster:
            print(point)
        print("---------------------")


def actual_clustering(initial_points):
    # returns what the actual clustering of the dataset is
    unique_clustering = list()
    for point in initial_points:
        exists = False
        if len(unique_clustering) == 0:
            unique_clustering.append(point[len(point) - 1])
            continue
        for cluster in unique_clustering:
            if point[len(point) - 1] == cluster:
                exists = True
        if not exists:
            unique_clustering.append(point[len(point) - 1])
    clustering = list()
    for cluster in unique_clustering:
        new_cluster = list()
        for point in initial_points:
            if point[len(point) - 1] == cluster:
                new_cluster.append(point)
        clustering.append(new_cluster)
    return clustering


def test_run():
    # this function runs all combinations of datasets and clustering algorithms and prints the output to out.txt as well as a tableof the hamming distances in output.csv
    sys.stdout = open("out.txt", "w")

    datasets = [['iris', 3, 'Iris'], ['glass', 6, 'Glass'], [
        'haberman', 2, 'Haberman']]
    results = [['Dataset', 'Single Linkage', 'Average Linkage', 'Lloyds n=1',
                'Lloyds n=100', 'Kmeans++ n=1', 'Kmeans++ n=100', 'naive Kmeans++ n=1', 'naive Kmeans++ n=100']]
    for dataset in datasets:
        print('Dataset: ', dataset[2])
        row = [dataset[2]]

        print('Algorithm: ', results[0][1])
        row.append(single_linkage(dataset[0], dataset[1]))

        print('Algorithm: ', results[0][2])
        row.append(average_linkage(dataset[0], dataset[1]))

        print('Algorithm: ', results[0][3])
        row.append(run_kmeans(dataset[0], dataset[1], 'lloyds', 1))

        print('Algorithm: ', results[0][4])
        row.append(run_kmeans(dataset[0], dataset[1], 'lloyds', 100))

        print('Algorithm: ', results[0][5])
        row.append(run_kmeans(dataset[0], dataset[1], 'kmeanspp', 1))

        print('Algorithm: ', results[0][6])
        row.append(run_kmeans(dataset[0], dataset[1], 'kmeanspp', 100))

        print('Algorithm: ', results[0][7])
        row.append(run_kmeans(dataset[0], dataset[1], 'naivekmeanspp', 1))

        print('Algorithm: ', results[0][8])
        row.append(run_kmeans(dataset[0], dataset[1], 'naivekmeanspp', 100))

        results.append(row)
    df = pd.DataFrame(results)
    df.to_csv('output.csv', index=False)
    print(df)
    sys.stdout.close()


test_run()
#run_kmeans('haberman', 2, 'kmeanspp', 1)
