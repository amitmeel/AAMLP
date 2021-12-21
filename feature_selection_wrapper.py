""" This module is used to create a wrapper for the feature selection"""

from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile

# create a wrapper to select univariate features
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

# if __name__=='__main__':
#     ufs = UnivariateFeatureSelction(
#         n_features=0.1,
#         problem_type="regression",
#         scoring="f_regression"
#     )
#     ufs.fit(X, y)
#     X_transformed = ufs.transform(X)

