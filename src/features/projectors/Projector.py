# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 15:19:49 2015

@author: pvrancx
"""

from Feature import Feature
import numpy as np


class Projector(Feature):
    normalized = False

    def __init__(self, normalize=False,
                 limits=np.array([[0, 1], [0, 1]]), bias=False):
        self.limits = limits
        self.ranges = limits[:, 1] - limits[:, 0]
        self.normalized = normalize
        self.bias = bias

    def project(self, state):
        phi = self.phi(state)
        if self.normalized:
            phi = phi / np.sum(phi)
        if self.bias:
            phi = np.r_[phi, [1.]]
        return phi

    def normalize_state(self, state):
        return (state - self.limits[:, 0]) / self.ranges

    ''' Indicates if projector supports returning nonzero indices/values only
        Returns: boolean
    '''

    def supports_sparse(self):
        return False
