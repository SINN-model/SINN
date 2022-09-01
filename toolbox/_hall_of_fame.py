#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle
import numpy as np


class HallOfFame:
    def __init__(self, capacity):
        self.capacity = capacity
        self.models = {}

    @staticmethod
    def _array(keys):
        return np.fromiter(keys, dtype=np.float64)

    def add(self, model, loss):
        models = self.models
        if (len(models) < self.capacity or
                loss < np.max(self._array(models.keys()))):
            models[loss] = pickle.loads(pickle.dumps(model))
        if len(models) > self.capacity:
            models.pop(np.max(self._array(models.keys())))

        self.models = {
            loss: self.models[loss]
            for loss in np.sort(self._array(models.keys()))
        }

    def __getitem__(self, i):
        return self.models[self._array(self.models.keys())[i]]
