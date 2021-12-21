""" This module is used to create a wrapper for the feature selection"""

import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn import linear_model
from sklearn import metrics
from sklearn.datasets import make_classification

# create a wrapper to select univariate features
"""
Univariate feature selection is nothing but a scoring of each feature
against a given target. Mutual information, ANOVA F-test and chi2 are
some of the most popular methods for univariate feature selection
"""
class UnivariateFeatureSelectionWrapper:
    """
    This class is used to create a wrapper for the feature selection for univariate features
    """

    def __init__(self, method, **kwargs):
        """
        This method is used to initialize the class
        :param method: the method used to select the features
        :param kwargs: the arguments for the method
        """
        self.method = method
        self.kwargs = kwargs

    def fit(self, X, y):
        """
        This method is used to fit the wrapper
        :param X: the features
        :param y: the labels
        :return: the wrapper
        """
        if self.method == 'chi2':
            self.selector = SelectKBest(chi2, k=self.kwargs['k'])
        elif self.method == 'f_classif':
            self.selector = SelectKBest(f_classif, k=self.kwargs['k'])
        elif self.method == 'f_regression':
            self.selector = SelectKBest(f_regression, k=self.kwargs['k'])
        elif self.method == 'mutual_info_classif':
            self.selector = SelectKBest(mutual_info_classif, k=self.kwargs['k'])
        elif self.method == 'mutual_info_regression':
            self.selector = SelectKBest(mutual_info_regression, k=self.kwargs['k'])
        elif self.method == 'percentile':
            self.selector = SelectPercentile(percentile=self.kwargs['percentile'])
        else:
            raise ValueError('Invalid method: {}'.format(self.method))
        self.selector.fit(X, y)
        return self

    def transform(self, X):
        """
        This method is used to transform the features
        :param X: the features
        :return: the transformed features
        """
        return self.selector.transform(X)

    def fit_transform(self, X, y):
        """
        This method is used to fit and transform the features
        :param X: the features
        :param y: the labels
        :return: the transformed features
        """
        self.fit(X, y)
        return self.transform(X)



class UnivariateFeatureSelction:
    def __init__(self, n_features, problem_type, scoring):
        """
        Custom univariate feature selection wrapper on
        different univariate feature selection models from
        scikit-learn.
        :param n_features: SelectPercentile if float else SelectKBest
        :param problem_type: classification or regression
        :param scoring: scoring function, string
        """
        # for a given problem type, there are only
        # a few valid scoring methods
        # you can extend this with your own custom
        # methods if you wish
        if problem_type == "classification":
            valid_scoring = {
            "f_classif": f_classif,
            "chi2": chi2,
            "mutual_info_classif": mutual_info_classif
            }
        else:
            valid_scoring = {
            "f_regression": f_regression,
            "mutual_info_regression": mutual_info_regression
            }
        # raise exception if we do not have a valid scoring method
        if scoring not in valid_scoring:
            raise Exception("Invalid scoring function")
        # if n_features is int, we use selectkbest
        # if n_features is float, we use selectpercentile
        # please note that it is int in both cases in sklearn
        if isinstance(n_features, int):
            self.selection = SelectKBest(
            valid_scoring[scoring],
            k=n_features
            )
        elif isinstance(n_features, float):
            self.selection = SelectPercentile(
            valid_scoring[scoring],
            percentile=int(n_features * 100)
            )
        else:
            raise Exception("Invalid type of features")

    def fit(self, X, y):
        """
        Fit the selection model
        :param X: data
        :param y: labels
        :return: self
        """
        self.selection.fit(X, y)
        return self
    
    def transform(self, X):
        """
        Transform the data
        :param X: data
        :return: transformed data
        """
        return self.selection.transform(X)

    def fit_transform(self, X, y):
        """
        Fit and transform the data
        :param X: data
        :param y: labels
        :return: transformed data
        """
        self.fit(X, y)
        return self.transform(X)

## greedy feature selection wrapper
"""
The simplest form of feature selection that uses a model for selection is known as
greedy feature selection. In greedy feature selection, the first step is to choose a
model. The second step is to select a loss/scoring function. And the third and final
step is to iteratively evaluate each feature and add it to the list of “good” features if
it improves loss/score
This feature selection process
will fit a given model each time it evaluates a feature. The computational cost
associated with this kind of method is very high.
"""
class GreedyFeatureSelection:
    """
    A simple and custom class for greedy feature selection.
    You will need to modify it quite a bit to make it suitable
    for your dataset.
    """
    def evaluate_score(self, X, y):
        """
        This function evaluates model on data and returns
        Area Under ROC Curve (AUC)
        NOTE: We fit the data and calculate AUC on same data.
        WE ARE OVERFITTING HERE.
        But this is also a way to achieve greedy selection.
        k-fold will take k times longer.
        If you want to implement it in really correct way,
        calculate OOF AUC and return mean AUC over k folds.
        This requires only a few lines of change and has been
        shown a few times in this book.
        :param X: training data
        :param y: targets
        :return: overfitted area under the roc curve
        """
        # fit the logistic regression model,
        # and calculate AUC on same data
        # again: BEWARE
        # you can choose any model that suits your data
        model = linear_model.LogisticRegression()
        model.fit(X, y)
        predictions = model.predict_proba(X)[:, 1]
        auc = metrics.roc_auc_score(y, predictions)

        return auc
    
    def _feature_selection(self, X, y):
        """
        This function does the actual greedy selection
        :param X: data, numpy array
        :param y: targets, numpy array
        :return: (best scores, best features)
        """
        # initialize good features list
        # and best scores to keep track of both
        good_features = []
        best_scores = []
        # calculate the number of features
        num_features = X.shape[1]
        # infinite loop
        while True:
            # initialize best feature and score of this loop
            this_feature = None
            best_score = 0
            # loop over all features
            for feature in range(num_features):
                # if feature is already in good features,
                # skip this for loop
                if feature in good_features:
                    continue
                # selected features are all good features till now
                # and current feature
                selected_features = good_features + [feature]
                # remove all other features from data
                xtrain = X[:, selected_features]
                # calculate the score, in our case, AUC
                score = self.evaluate_score(xtrain, y)
                # if score is greater than the best score
                # of this loop, change best score and best feature
                if score > best_score:
                    this_feature = feature
                    best_score = score
                # if we have selected a feature, add it
                # to the good feature list and update best scores list
                if this_feature != None:
                    good_features.append(this_feature)
                    best_scores.append(best_score)
                # if we didnt improve during the previous round,
                # exit the while loop
                if len(best_scores) > 2:
                    if best_scores[-1] < best_scores[-2]:
                        break
        # return best scores and good features
        # why do we remove the last data point?
        return best_scores[:-1], good_features[:-1]

    def __call__(self, X, y):
        """
        Call function will call the class on a set of arguments
        """
        # select features, return scores and selected indices
        scores, features = self._feature_selection(X, y)
        # transform data with selected features
        return X[:, features], scores


# if __name__=='__main__':
#     ufs = UnivariateFeatureSelction(
#         n_features=0.1,
#         problem_type="regression",
#         scoring="f_regression"
#     )
#     ufs.fit(X, y)
#     X_transformed = ufs.transform(X)

# if __name__ == "__main__":
#     # generate binary classification data
#     X, y = make_classification(n_samples=1000, n_features=100)
#     # transform data by greedy feature selection
#     X_transformed, scores = GreedyFeatureSelection()(X, y)
