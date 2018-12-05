import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

animal_array
animal_dataframe = pd.DataFrame(toy_data, columns = column_names)

animal_train, animal_test= train_test_split(toy_dataframe, test_size=0.2, random_state=42)
toy_train,toy_test = toy_train.reset_index(drop = True), toy_test.reset_index(drop = True)


