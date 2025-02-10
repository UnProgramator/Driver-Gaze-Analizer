from sklearn.cluster import KMeans
import numpy as np


a = np.asarray(
    [
        [[1,2],[2,4]],
        [[7,2],[2,8]],
        [[-1,7],[2,3]],
        [[10,2],[29,4]],
        [[7,20],[22,87]],
        [[-8,7],[-2,-3]]

    ]
    )

print(a)

kmean = KMeans(n_clusters=3, random_state=0, n_init="auto")

m = kmean.fit(a)

print(m.labels_)