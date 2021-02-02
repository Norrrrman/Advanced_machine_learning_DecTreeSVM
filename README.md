# Advanced_machine_learning_DecTreeSVM

1. Training and Visualizing a Decision Tree
1.1. First, let's load the iris dataset from sci-kit learn library. Plot the data set.
1.2. Use the function train_test_split(X,y,test_size=0.3, random_state=42) and
DecisionTreeClassifier() to train the dataset using the Decision Tree Model.
Compute the accuracy score by writing your own function.
2. Decision Trees parameters
There are many hyperparameters that a decision tree classifier has. Analyze the meaning
of theses parameters.
Will you use gini index or entropy index. Try the both and explain the results.
3. Visualize and save the Tree
We can also save the decision tree which our model has built. You need to have Graph Viz and
pydotplus installed in your system.
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(model, out_file=dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")
from IPython.display import Image
Image(filename="./images/tree.png")
Another way to visualize decision trees is to install the dtreeviz package :
!pip install dtreeviz
Use the function dtreeviz to plot the tree.
You can plot the decision boundary on the scatter of points.
Create the function plot_decision_boundary(clf, x, y) where clf is the Decision Tree classifier, X
is the dataset and y is the target.
4. Estimating Class Probabilities
To estimate the probability of an instance belongs to a class, you can use predict_proba, to
determine the class that an instance will be assigned to use predict.
5. Regression
Decision trees can be used for regression tasks too. Instead of predicting a class, in regression
tasks, the aim is to predict a numeric value.
- Load the diabetes dataset and use the DecisionTreeRegressor function to predict the diabetes
evolution (target variable).
- Compute the MSE error and compare it if using the Linear Regression. Conclude.
7. Support Vector Machine
On the iris dataset use the SVC function and train_test_split to classify and predict the class on
test data. Compute the accuracy and compare it to those obtained using Decision Tree.
Normalize the data using StandardScaler() and apply again the SVC. How the accuracy
changes?
8. Kernel trick.
The kernel parameter of the SVC function allows to define different kernels functions as : Radial
Basis Function Kernel, Polynomial Kernel, Sigmoid kernel, etc.
Try several of them by comparing the obtained accuracy score.
