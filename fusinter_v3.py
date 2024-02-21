import FUSINTER_v3_pybind
import numpy as np
class FUSINTERDISCRETIZER:
    def __init__(self, alpha, lam):
        self.discretizer = FUSINTER_v3_pybind.FUSINTERDiscretizer(alpha, lam)
        self.splits = None
    def fit(self, x, y):
        self.splits = self.discretizer.fit(x,y)
        return self.splits

    def transform(self,x ):
        return np.digitize(x, self.splits)