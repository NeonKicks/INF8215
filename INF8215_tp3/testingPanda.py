import numpy as np
import pandas as pd

from random import random
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from preprocessing import TransformationWrapper
from preprocessing import LabelEncoderP
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer


# Redistribute "Unknown" sexes uniformly across the four known categories
def parse_unknown_sex(text):
    if text == "Unknown":
        rand = random()
        if rand <= 0.141578:
            return "Intact Female"
        elif rand <= 0.283680:
            return "Intact Male"
        elif rand <= 0.623882:
            return "Spayed Female"
        else:
            return "Neutered Male"

def parse_sex(text):
    _, sex = text.split(" ")
    return sex

def parse_fixed(text):
    fixed, _ = text.split(" ")
    if fixed == "Intact":
        return fixed
    elif fixed == "Spayed" or fixed == "Neutered":
        return "Fixed"

def parse_age(text):
    if text == "Unknown years":
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
        print("Neither, days, weeks, months, nor years")

def parse_mix(text):
    # Check if 'Breed' contains a '/'
    if len([text]) != len(text.split("/")):
        return "mix"
    else:
        # parse words composing breed
        parsed_text = text.split(" ")
        for word in parsed_text:
            if word == "Mix" or word == "mix":
                return "Mix"
        
        return "Pure"
    
def parse_breed(text):
    # parse words composing breed
    parsed_text = text.split(" ")
    


############## Pipelines ####################




##### Component pipeline #######
pipeline_sex = Pipeline([
        ("sex", TransformationWrapper( transformation = parse_sex)),
        ("encode", LabelEncoderP()),
    ])
pipeline_fixed = Pipeline([
        ("fixed", TransformationWrapper( transformation = parse_fixed)),
        ("encode", LabelEncoderP()),
    ])

##### Column modifiers #######
pipeline_age = Pipeline([
        ("age_imputer", SimpleImputer(strategy = 'constant', fill_value = 'Unknown years')),
        ("age_converter", TransformationWrapper(transformation = parse_age)),
        ("fillna", SimpleImputer(missing_values = 0.0, strategy = 'mean') ),
        ("scaler", StandardScaler()),
    ])
pipeline_type = Pipeline([
        ("encode", LabelEncoderP()),
    ])
pipeline_sex_state = Pipeline([
        ("sex_imputer", TransformationWrapper(transformation = parse_unknown_sex)),
        ('feats', FeatureUnion([
            ('sex', pipeline_sex), 
            ('fixed', pipeline_fixed),
        ])),
    ])
#pipeline_breed = Pipeline()



####### Full pipeline #########
full_pipeline = ColumnTransformer([
        ("age", pipeline_age, ["AgeuponOutcome"]),
        ("type", pipeline_type, ["AnimalType"]),
        ("sex", pipeline_sex_state, ["SexuponOutcome"]),
        #("breed", pipeline_breed, ["Breed"]),
    ])



# Fetch data
x_train = pd.read_csv("./data/train.csv", header=0)
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
print(x_train["Breed"].value_counts()/len(x_train))



#full_pipeline_columns = ["age","type","sex","fixed"]
#X_train = pd.DataFrame(full_pipeline.fit_transform(x_train), columns = full_pipeline_columns)
#print(X_train.head())



"""
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
"""





