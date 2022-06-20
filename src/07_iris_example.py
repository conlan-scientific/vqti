
import pandas as pd
from sklearn import datasets

iris_meta = datasets.load_iris()
X = pd.DataFrame(iris_meta['data'], columns=iris_meta['feature_names'])
y = pd.Series(iris_meta['target'], name='target')



"""
The fundamental problem of machine learning
-------------------------------------------
Given a vector y ...
Build a matrix X ...
Determine f() to minimize epsilon. 

y = f(X) + epsilon^2

y can be called the target, the label, the objective ...

y_hat is the predicted value of y based on X
"""

# from sklearn import svm
# classifier = svm.SVC(gamma=0.001, C=100)

from sklearn import tree
classifier = tree.DecisionTreeClassifier(max_depth=6)

classifier.fit(X, y)
y_hat = classifier.predict(X)
y_hat = pd.Series(y_hat, name='y_hat')
print(f'Accuracy: {100 * (y == y_hat).mean():.2f}%')


