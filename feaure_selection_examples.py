"""
Another greedy approach is known as recursive feature elimination (RFE). In the
previous method, we started with one feature and kept adding new features, but in
RFE, we start with all features and keep removing one feature in every iteration that
provides the least value to a given model. But how to do we know which feature
offers the least value? Well, if we use models like linear support vector machine
(SVM) or logistic regression, we get a coefficient for each feature which decides
the importance of the features. In case of any tree-based models, we get feature
importance in place of coefficients. In each iteration, we can eliminate the least
important feature and keep eliminating it until we reach the number of features
needed.
"""


## Example 1
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
# fetch a regression dataset
data = fetch_california_housing()
X = data["data"]
col_names = data["feature_names"]
y = data["target"]
# initialize the model
model = LinearRegression()
# initialize RFE
rfe = RFE(
estimator=model,
n_features_to_select=3
)
# fit RFE
rfe.fit(X, y)
# get the transformed data with
# selected columns
X_transformed = rfe.transform(X)
#===============================================================================

#   Example 2
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
# fetch a regression dataset
# in diabetes data we predict diabetes progression
# after one year based on some features
data = load_diabetes()
X = data["data"]
col_names = data["feature_names"]
y = data["target"]
# initialize the model
model = RandomForestRegressor()
# fit the model
model.fit(X, y)
importances = model.feature_importances_
idxs = np.argsort(importances)
plt.title('Feature Importances')
plt.barh(range(len(idxs)), importances[idxs], align='center')
plt.yticks(range(len(idxs)), [col_names[i] for i in idxs])
plt.xlabel('Random Forest Feature Importance')
plt.show()
#===============================================================================

#   Example 3 Using scikit learn SelectFromModel
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
# fetch a regression dataset
# in diabetes data we predict diabetes progression
# after one year based on some features
data = load_diabetes()
X = data["data"]
col_names = data["feature_names"]
y = data["target"]
# initialize the model
model = RandomForestRegressor()
# select from the model
sfm = SelectFromModel(estimator=model)
X_transformed = sfm.fit_transform(X, y)
# see which features were selected
support = sfm.get_support()
# get feature names
print([
x for x, y in zip(col_names, support) if y == True
])