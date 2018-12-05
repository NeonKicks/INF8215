from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class SoftmaxClassifier(BaseEstimator, ClassifierMixin):  
    """A softmax classifier"""

    def __init__(self, lr = 0.1, alpha = 100, n_epochs = 1000, eps = 1.0e-5,threshold = 1.0e-10 , regularization = True, early_stopping = True):
       
        """
            self.lr : the learning rate for weights update during gradient descent
            self.alpha: the regularization coefficient 
            self.n_epochs: the number of iterations
            self.eps: the threshold to keep probabilities in range [self.eps;1.-self.eps]
            self.regularization: Enables the regularization, help to prevent overfitting
            self.threshold: Used for early stopping, if the difference between losses during 
                            two consecutive epochs is lower than self.threshold, then we stop the algorithm
            self.early_stopping: enables early stopping to prevent overfitting
        """

        self.lr = lr 
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.eps = eps
        self.regularization = regularization
        self.threshold = threshold
        self.early_stopping = early_stopping
        ### This line is strictly for testing purposes
        self.pause = False


    """
        Public methods, can be called by the user
        To create a custom estimator in sklearn, we need to define the following methods:
        * fit
        * predict
        * predict_proba
        * fit_predict        
        * score
    """


    """
        In:
        X : the set of examples of shape nb_example * self.nb_features
        y: the target classes of shape nb_example *  1

        Do:
        Initialize model parameters: self.theta_
        Create X_bias i.e. add a column of 1. to X , for the bias term
        For each epoch
            compute the probabilities
            compute the loss
            compute the gradient
            update the weights
            store the loss
        Test for early stopping

        Out:
        self, in sklearn the fit method returns the object itself


    """

    def fit(self, X, y=None):
        
        prev_loss = np.inf
        self.losses_ = []

        self.nb_feature = X.shape[1]
        self.nb_classes = len(np.unique(y))

        

        X_bias = np.ones((X.shape[0], X.shape[1]+1))
        X_bias[:,:-1] = X



        self.theta_= np.random.rand(self.nb_feature + 1, self.nb_classes)

        
        

        for epoch in range( self.n_epochs):
            # Do matrix product of X_bias and self.theta_
            logits = np.dot(X_bias, self.theta_)
            
            # Calculate probabilities with softmax function
            probabilities = self._softmax(logits)
           
            # Calculate log loss
            loss = self._cost_function(probabilities,y)

            # Adjut theta
            self.theta_ = self.theta_- (np.multiply(self.lr, self._get_gradient(X_bias, y, probabilities)))

            # Function to allow pausing between interations
            if self.pause == True:
                    print("pause next loop? y/n")
                    answer = input()
                    if(answer == "n"):
                            self.pause = False

            # Add log loss to list of log losses
            self.losses_.append(loss)

    
            # If it is setup for early stopping, stop when log loss crosses threshold
            if self.early_stopping == True:
                    if(len(self.losses_) > 1):
                        #if abs(self.losses_[-2] - self.losses_[-1]) < self.threshold:
                        if self.losses_[-2] - self.losses_[-1] < self.threshold:
                            print("early stopping at epoch " + str(epoch))
                            break

        return self

    

   
    

    """
        In: 
        X without bias

        Do:
        Add bias term to X
        Compute the logits for X
        Compute the probabilities using softmax

        Out:
        Predicted probabilities
    """

    def predict_proba(self, X, y=None):
        try:
            getattr(self, "theta_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        
        X_bias = np.ones((X.shape[0], X.shape[1]+1))
        X_bias[:,:-1] = X

        z = np.dot(X_bias, self.theta_)

        return self._softmax(z)

        """
        In: 
        X without bias

        Do:
        Add bias term to X
        Compute the logits for X
        Compute the probabilities using softmax
        Predict the classes

        Out:
        Predicted classes
    """

    
    def predict(self, X, y=None):
        try:
            getattr(self, "theta_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        pass

        probabilities = self.predict_proba(X,y)
        return np.argmax(probabilities, axis=1)

    

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X,y)


    """
        In : 
        X set of examples (without bias term)
        y the true labels

        Do:
            predict probabilities for X
            Compute the log loss without the regularization term

        Out:
        log loss between prediction and true labels

    """    

    def score(self, X, y=None):
        self.regulariation = False
        predictions = self.predict(X)

        return self._cost_function(self.predict_proba(X), y=y)
        

    """
        Private methods, their names begin with an underscore
    """

    """
        In :
        y without one hot encoding
        probabilities computed with softmax

        Do:
        One-hot encode y
        Ensure that probabilities are not equal to either 0. or 1. using self.eps
        Compute log_loss
        If self.regularization, compute l2 regularization term
        Ensure that probabilities are not equal to either 0. or 1. using self.eps

        Out:
        cost (real number)
    """
    
    def _cost_function(self,probabilities, y ): 
        one_hot_y = self._one_hot(y)

        probabilities[probabilities == 0.] = self.eps
        probabilities[probabilities == 1.] = 1. - self.eps



        # TODO: Verify if it should be log(l2) or just l2 being added
        if(self.regularization == True):
            l2 = np.multiply(float(self.alpha)/float(probabilities.shape[0]), np.sum(np.square(self.theta_)))
        else:
            l2 = 0

        log_loss = np.multiply(-(1./probabilities.shape[0]), np.sum(np.multiply(one_hot_y, np.log(probabilities))))

        return log_loss + l2
    

    
    """
        In :
        Target y: nb_examples * 1

        Do:
        One hot-encode y
        [1,1,2,3,1] --> [[1,0,0],
                         [1,0,0],
                         [0,1,0],
                         [0,0,1],
                         [1,0,0]]
        Out:
        y one-hot encoded
    """

    def _one_hot(self,y):
        # Initializing a 2d array with len(y) rows and nb_classes columns
        y_one_hot = np.zeros((len(y),self.nb_classes))

        y_one_hot[np.arange(len(y)),y] = 1
        return y_one_hot

    


    """
        In :
        Logits: self.nb_examples * self.nb_classes

        Do:
        Compute softmax on logits

        Out:
        Probabilities
    """
    
    def _softmax(self,z):
        exponentials = np.exp(z)
        totals = np.sum(exponentials,axis=1)

        totals = np.divide(np.ones(totals.shape), totals)

        p = (exponentials.T * totals).T

        return p
    

    """
        In:
        X with bias
        y without one hot encoding
        probabilities resulting of the softmax step

        Do:
        One-hot encode y
        Compute gradients
        If self.regularization add l2 regularization term

        Out:
        Gradient

    """

    def _get_gradient(self,X,y, probas):
        yohe = self._one_hot(y)
        
        if (self.regularization == True):
            l2 = np.multiply(float(self.alpha)/(float(X.shape[0])), (self.theta_))
        else:
            l2 = 0

        gradient = np.multiply(1./(X.shape[0]), np.dot(X.T, np.subtract(probas,yohe)))
        return gradient + l2
    
    
