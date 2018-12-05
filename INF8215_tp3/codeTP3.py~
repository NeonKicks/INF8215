#!/usr/bin/python


# Ce fichier sert à faciliter la mise en commun avec git


####### Partie 1. Softmax Regression ##############


# Question 1 (1 point)
#  Implement the onehot method in SoftmaxClassifier.py




# Question 2 (1 point)
#  In the fit function in SoftmaxClassifier.py instantiate X_bias and
#  initialize (teta) randomly. (line 74)




# Question 3 (1 point)
#  Implement _softmax method in SoftmaxClassifier.py






# Question 4 (1 point)
#  Using the _ softmax function of question 3, implement the predict_proba and
#  predict methods in SoftmaxClassifier.py





# Question 5 (1 point)
#  Implement the _ cost_function method in SoftmaxClassifier.py by taking into
#  account the implementation detail (self.eps variable) and use it to
#  calculate the loss variable in the fit method (line 84)






# Question 6 (1 point)
#  Implement getgradient method in SoftmaxClassifier.py





# Question 7 (1 point)
#  Update self.theta_ in the fit method in SoftmaxClassifier.py (line 85)






# Question 8 (1 point)
#  Modify the methods _ get_gradient and _ cost_function to take into account
#  the regularization when the boolean self.regularization is true in
#  SoftmaxClassifier.py







# Question 9 (1 point)
#  The regularization term is used only during training. When one wants to
#  evaluate the performance of the model after training, one uses the
#  non-regulated cost function.  Implement the score function that evaluates
#  the quality of the prediction after training in SoftmaxClassifier.py









# Question 10 (1 point)
#  Finish implementing the fit function by adding the early stopping mechanism
#  when the self.early_stopping boolean is true. The threshold is given by the
#  self.threshold variable .








# Testing the solution
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load dataset
data,target =load_iris().data,load_iris().target

# split data in train/test sets
X_train, X_test, y_train, y_test = train_test_split( data, target, test_size=0.33, random_state=42)

# standardize columns using normal distribution
# fit on X_train and not on X_test to avoid Data Leakage
s = StandardScaler()
X_train = s.fit_transform(X_train)
X_test = s.transform(X_test)
"""




"""
from SoftmaxClassifier import SoftmaxClassifier

# import the custom classifier
cl = SoftmaxClassifier()

# train on X_train and not on X_test to avoid overfitting
train_p = cl.fit_predict(X_train,y_train)
test_p = cl.predict(X_test)
"""


"""
from sklearn.metrics import precision_recall_fscore_support

# display precision, recall and f1-score on train/test set
print("train : "+ str(precision_recall_fscore_support(y_train, train_p,average = "macro")))
print("test : "+ str(precision_recall_fscore_support(y_test, test_p,average = "macro")))
"""


"""
# import matplotlib.pyplot as plt

# plt.plot(cl.losses_)
# plt.show()
"""



########### Partie 2. Data preprocessing #############

# Dataset
"""
import pandas as pd

PATH = "data/"
X_train = pd.read_csv(PATH + "train.csv")
X_test = pd.read_csv(PATH + "test.csv")
"""

"""
X_train = X_train.drop(columns = ["OutcomeSubtype","AnimalID"])
X_test = X_test.drop(columns = ["ID"])
"""


"""
X_train, y_train = X_train.drop(columns = ["OutcomeType"]),X_train["OutcomeType"]
"""


"""
X_train.head()
"""


"""
X_test.head()
"""


"""
y_train.head()
"""


"""
X_train1 = pd.read_csv("data/train_preprocessed.csv")
X_test1 = pd.read_csv("data/test_preprocessed.csv")
"""


"""
X_train1.head()
"""


"""
X_train = X_train.drop(columns = ["Color","Name","DateTime"])
X_test = X_test.drop(columns = ["Color","Name","DateTime"])
"""


"""
X_train.head()
"""




# Question 11: AgeuponOutcome (1 point)
#  ...

# Question 12: AnimalType (1 point)
#  ...

# Question 13: SexuponOutcome (1 point)
#  ...

# Question 14: Breed (1 point)
#  ...


# Pipeline

"""
from preprocessing import TransformationWrapper
from preprocessing import LabelEncoderP
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer

# pipeline_color = Pipeline([
#     ("name", Transformer()),
# ])


# full_pipeline = ColumnTransformer([
#         ("color", pipeline_color, ["Color"]),


#     ])

"""


# Run the pipeline
"""
# column_names = []
# X_train_prepared = pd.DataFrame(full_pipeline.fit_transform(X_train),columns = columns)
# X_test_prepared = pd.DataFrame(full_pipeline.fit_transform(X_test),columns = columns)
"""


# Concatenate both parts of the dataset
"""
# X_train = pd.concat([X_train1,X_train_prepared], axis = 1)
# X_test = pd.concat([X_test1,X_test_prepared], axis = 1)
"""




####### Partie 3. Model Selection (2 points) ##########
"""
from sklearn.preprocessing import LabelEncoder
target_label = LabelEncoder()
y_train_label = target_label.fit_transform(y_train)
print(target_label.classes_)
"""


# Bonus 2: StratifiedKFold (1 point)
#  By observing the class distribution of the target attribute (using the
#  pandas visualization functions), justify the use of the sklearn
#  StratifiedKFold object for division of the training set when doing
#  cross-validation instead of a pure random method.






# Question 16: (1 point)
#  Choose at least two models allowing the multiclass classification on sklearn
#  in addition to the model implemented in the first part of the TP.  Complete
#  the compare function that performs the crossvalidation for different models
#  and different metrics, and returns the list of averages and standard
#  deviations for each of the metrics, for each of the models.  Based on the
#  different metrics, conclude on the best performing model.  Evaluate the
#  models for the different metrics proposed: log loss: this is the kaggle
#  evaluation metric for this dataset precision: corresponds to the quality of
#  the prediction, the number of classes correctly predicted by the total
#  prediction number recall: the number of elements belonging to a class,
#  identified as such, divided by the total number of elements of that class.
#  f-score: an average of accuracy and recall Note: Precision and recall are
#  two complementary measures for evaluating a multi-class classification
#  model.  In the case of a binary classification with an important target
#  class imbalance, (90% / 10%), evaluating the classification result with
#  accuracy (number of correct predictions divided by the total number of
#  predictions), a very good score (90% accuracy) can be obtained by choosing
#  to systematically predict the majority class.  In such a case, the precision
#  would be high in the same way, but the recall would be very low, indicating
#  the mediocrity of our model.




"""
def compare(models,X_train,y_train,nb_runs):
    losses = []

    return losses
"""



"""
from SoftmaxClassifier import SoftmaxClassifier

nb_run = 3

models = [
    SoftmaxClassifier(),
]

scoring = ['neg_log_loss', 'precision_macro','recall_macro','f1_macro']

compare(models,X_train,y_train_label,nb_run,scoring)
"""



# Question 17: Confusion matrix (0.5 point)
#  The confusion matrix A is such that Aij represents the number of examples of
#  class i classified as belonging to class j.  Train the selected model on the
#  entire training set. Using the confusion matrix and class distribution,
#  analyze in more detail the performance of the chosen model and justify them.

"""
# Train selected model

selected_model =
y_pred = selected_model.fit_predict(X_train,y_train_label)
"""

# Confusion matrix
"""
from sklearn.metrics import confusion_matrix
pd.DataFrame(confusion_matrix(y_train_label, y_pred), columns = target_label.classes_, index = target_label.classes_)
"""


# Target class distribution
"""
import matplotlib.pyplot as plt 
print(target_label.classes_)
pd.Series(y_train_label).hist()
"""




# Bonus 3: Hyper-parameters optimization (1 point)
#  Hyper-parameters are the parameters set before the learning phase. To
#  optimize the performance of the model, we can select the best
#  hyper-parameters.  Using sklearn, optimize the hyper-parameters of the model
#  you have selected and show that the performance has been improved. For
#  example, you can use: GridSearchCV Finally, make the prediction on the test
#  set and give your results when submitting the lab.  Optional: You can submit
#  your results on kaggle and note your performance in terms of log loss.



"""
# best_model = 
# pred_test = pd.Series(best_model.transform(X_test))
# pred_test.to_csv("test_prediction.csv",index = False)
"""
