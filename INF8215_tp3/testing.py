import numpy as np

def _one_hot(y):
        # Initializing a 2d array with len(y) rows and nb_classes columns
        y_one_hot = np.zeros((len(y),3))

        for idx, val in enumerate(y):
            y_one_hot[idx][val-1] = 1

        return y_one_hot


y = [1,1,2,3,1]

print(_one_hot(y))