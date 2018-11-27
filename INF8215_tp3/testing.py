import numpy as np

def _one_hot(y):
        # Initializing a 2d array with len(y) rows and nb_classes columns
        y_one_hot = np.zeros((len(y),4))

        y_one_hot[np.arange(len(y)),y] = 1 
        """ 
        for idx, val in enumerate(y):
            y_one_hot[idx][val-1] = 1
        """

        return y_one_hot

y = [1,1,2,3,1,0]

print(_one_hot(y))
