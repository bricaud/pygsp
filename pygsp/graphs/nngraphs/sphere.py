# -*- coding: utf-8 -*-

import numpy as np

from pygsp.graphs import NNGraph  # prevent circular import in Python < 3.5


class Sphere(NNGraph):
    pass

class SphereAngles(Sphere):
    pass

class SphereRandomUniform(Sphere):
    r"""Spherical-shaped graph (NN-graph).

    Random uniform sampling of the d-dimensional sphere.

    Parameters
    ----------
    radius : float
        Radius of the sphere (default = 1)
    nb_pts : int
        Number of vertices (default = 300)
    nb_dim : int
        Dimension (default = 3)
    sampling : string
        Variance of the distance kernel (default = 'random')
        (Can now only be 'random')
    seed : int
        Seed for the random number generator (for reproducible graphs).

    References
    ----------
    http://mathworld.wolfram.com/HyperspherePointPicking.html
    Hicks, J. S. ad Wheeling, R. F. "An Efficient Method for Generating Uniformly Distributed Points on the Surface of an n-Dimensional Sphere." Comm. Assoc. Comput. Mach. 2, 13-15, 1959.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Sphere(nb_pts=100, seed=42)
    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(121)
    >>> ax2 = fig.add_subplot(122, projection='3d')
    >>> _ = ax1.spy(G.W, markersize=1.5)
    >>> _ = _ = G.plot(ax=ax2)

    """

    def __init__(self,
                 n_points=300,
                 n_dimensions=3,
                 radius=1,
                 n_neighbors=10,
                 seed=None,
                 **kwargs):

        self.n_dimensions = n_dimensions
        self.radius = radius
        self.seed = seed

        rs = np.random.RandomState(seed)
        points = rs.normal(0, 1, (n_points, n_dimensions))
        points /= np.linalg.norm(points, axis=1)[:, np.newaxis]

        plotting = {
            'vertex_size': 80,
        }

        super(SphereRandomUniform, self).__init__(points, k=n_neighbors, center=False, rescale=False,
                                     plotting=plotting, **kwargs)

    def _get_extra_repr(self):
        attrs = {'radius': '{:.2f}'.format(self.radius),
                 'n_dimensions': self.n_dimensions,
                 'seed': self.seed}
        attrs.update(super(SphereRandomUniform, self)._get_extra_repr())
        return attrs
