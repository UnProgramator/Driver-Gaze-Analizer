from typing_extensions import override
from sklearn.cluster import KMeans
import numpy as np

def map_e(e):
    return ord(e)

def map_l(l):
    return list(map(map_e, l))

car = [['a', 'c', 'c'], ['a', 'd', 'e'], ['a', 'f', 'e'], ['b', 'c', 'f'], ['b', 'd', 'a'], ['b', 'f', 'a']]

ar = list(map(map_l, car))

print(ar)

x = np.array(ar)

def dist(box_dims, centroid_box_dims):
    box_w, box_h = box_dims[..., 0], box_dims[..., 1]
    centroid_w, centroid_h = centroid_box_dims[..., 0], centroid_box_dims[..., 1]
    inter_w = np.minimum(box_w[..., np.newaxis], centroid_w[np.newaxis, ...])
    inter_h = np.minimum(box_h[..., np.newaxis], centroid_h[np.newaxis, ...])
    inter_area = inter_w * inter_h
    centroid_area = centroid_w * centroid_h
    box_area = box_w * box_h
    return inter_area / (
        centroid_area[np.newaxis, ...] + box_area[..., np.newaxis] - inter_area
    )

class MyClass(KMeans):
    def __init__(self,
        n_clusters=8,
        *,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm="lloyd",
    ):
        super().__init__(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            copy_x=copy_x,
            algorithm=algorithm
        )
   
    @override
    def transform(self, X):
        return dist(X, self.cluster_centers_)

kmean = MyClass(n_clusters=3, random_state=0, n_init="auto")

m = kmean.fit(ar)

print(m.labels_)