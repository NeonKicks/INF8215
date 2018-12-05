import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from preprocessing import TransformationWrapper
from preprocessing import LabelEncoderP
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer


def parse_sex(text):
    if text == "Unknown":
        return text
    else:
        _, sex = text.split(" ")
        return sex

def parse_fixed(text):
    if text == "Unknown":
        return text
    else:
        fixed, _ = text.split(" ")
        return fixed

def parse_age(text):
    if text == "Unknown Age":
        return 0.0

    number, resolution = text.split(" ")
    if "year" in resolution:
        return float(number) 
    elif "month" in resolution:
        return float(number) / 12.
    elif "week" in resolution:
        return float(number) / 52.
    elif "day" in resolution:
        return float(number) / 365.
    else:
        print("Neither, days, weekd, months, nor years")



# Fetch data
x_train = pd.read_csv("train.csv", header=0)
print(x_train.head())

# Remove unecessary columns
x_train = x_train.drop(columns = ["OutcomeSubtype","AnimalID"])
print(x_train.head())

# Extract specific column and save it in another dataframe
x_train, y_train = x_train.drop(columns = ["OutcomeType"]), x_train["OutcomeType"]
print(x_train.head())
print(y_train.head())

# Visualize data
print(x_train["AnimalType"].value_counts()/len(x_train))
print(x_train["SexuponOutcome"].value_counts()/len(x_train))






####### 'age' #######
# Fix missing data in AgeuponOutcome column
age_imputer = SimpleImputer(strategy = 'constant', fill_value = 'Unknown Age')
x_train["AgeuponOutcome"] = age_imputer.fit_transform(x_train["AgeuponOutcome"].values.reshape(-1,1))

# Reformat 'AgeuponOutcome'
age_train = x_train.apply(lambda row: pd.Series(  parse_age(row["AgeuponOutcome"])  ), axis = 1  )
age_train.columns = ["age"]

# Fix missing data in AgeuponOutcome column
age_imputer2 = SimpleImputer(missing_values=0.0, strategy='mean')
age_train["age"] = age_imputer2.fit_transform(age_train["age"].values.reshape(-1,1))





###### 'sex' #######
# Fix missing data in SexuponOutcome column
sextual_imputer = SimpleImputer(strategy = 'constant', fill_value = 'Unknown')
x_train["SexuponOutcome"] = sextual_imputer.fit_transform(x_train["SexuponOutcome"].values.reshape(-1,1))

# Split 'SexuponOutcome' into 'sex' and 'fixed'
sex_train = x_train.apply(lambda row: pd.Series(  parse_sex(row["SexuponOutcome"])  ), axis = 1  )
fixed_train = x_train.apply(lambda row: pd.Series(  parse_fixed(row["SexuponOutcome"])  ), axis = 1  )

sex_train.columns = ["sex"]
fixed_train.columns = ["fixed"]



new_columns = pd.concat([sex_train, fixed_train, age_train], axis = 1)
x_train = x_train.drop(columns = ["SexuponOutcome","AgeuponOutcome"])
x_train = pd.concat([x_train, pd.DataFrame(new_columns)], axis = 1)


print(x_train.loc[[2480]])

