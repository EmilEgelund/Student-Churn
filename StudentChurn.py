import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# 1 label, Churn, with 5 features. ID should be removed, because it arbitrarily assigned and holds no value.
col_names = ["Id", "Churn", "Line", "Grade", "Age", "Distance", "StudyGroup"]


data = pd.read_csv('StudentChurn.csv', delimiter=';', names=col_names, header=0)

# show the data
print ( data .describe( include = 'all' ))
print ( data .values)

print(data.columns)
print(data.shape)


x = data[ "Line" ]
y = data[ "Age" ]
plt.figure()
plt.scatter(x.values, y.values, color = 'black' , s = 20 )
#plt.show()

x = data[ "Grade" ]
y = data[ "Age" ]
plt.figure()
plt.scatter(x.values, y.values, color = 'black' , s = 20 )
#plt.show()

x = data[ "Line" ]
y = data[ "Grade" ]
plt.figure()
plt.scatter(x.values, y.values, color = 'black' , s = 20 )
#plt.show()

x = data[ "Grade" ]
y = data[ "Churn" ]
plt.figure()
plt.scatter(x.values, y.values, color = 'black' , s = 20 )
#plt.show()

# Converting categorial features to numerical
data['Churn'] = data['Churn'].replace(['Stopped'], 0.0)
data['Churn'] = data['Churn'].replace(['Completed'], 1.0)

data['Line'] = data['Line'].astype('category').cat.codes

# Drop rows where 'StudyGroup' is blank
data = data[data['StudyGroup'].notna()]

# Drop unused features.
data.drop('Id', axis=1, inplace=True)

#Copy 'Churn' into yvalues and delete from data.
yvalues = pd.DataFrame(dict(Churn=[]), dtype=int)
yvalues['Churn'] = data['Churn'].copy()
data.drop('Churn', axis=1, inplace=True)

# 80/20'ish split between training and testing.
#750/336 split yielded 84.82% accuracy, up 0.55%
xtrain = data.head(750)
xtest = data.tail(336)

ytrain = yvalues.head(750)
ytest = yvalues.tail(336)

# Scaling the data
scaler = StandardScaler()
scaler .fit(xtrain)
xtrain = scaler .transform(xtrain)
xtest = scaler .transform(xtest)

# Good ones, 84.27%
mlp_configs = [
    {'hidden_layer_sizes': (6, 6, 6, 6, 6), 'activation': 'relu', 'max_iter': 250, 'solver': 'adam',
     'learning_rate_init': 0.0005},
    {'hidden_layer_sizes': (8, 8, 8, 8), 'activation': 'tanh', 'max_iter': 300, 'solver': 'sgd',
     'learning_rate_init': 0.01},

]

mlp_configs2 = [
    {'hidden_layer_sizes': (12, 12, 12), 'activation': 'logistic', 'max_iter': 150, 'solver': 'lbfgs', 'learning_rate_init': 0.005},
    {'hidden_layer_sizes': (12, 12, 12), 'activation': 'logistic', 'max_iter': 150, 'solver': 'lbfgs', 'learning_rate_init': 0.01},
    {'hidden_layer_sizes': (12, 12, 12), 'activation': 'logistic', 'max_iter': 150, 'solver': 'lbfgs', 'learning_rate_init': 0.0025},
    {'hidden_layer_sizes': (10, 10, 10), 'activation': 'relu', 'max_iter': 200, 'solver': 'adam', 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (8, 8, 8, 8), 'activation': 'tanh', 'max_iter': 300, 'solver': 'sgd', 'learning_rate_init': 0.01},
    {'hidden_layer_sizes': (6, 6, 6, 6, 6), 'activation': 'relu', 'max_iter': 250, 'solver': 'adam', 'learning_rate_init': 0.0005},
    {'hidden_layer_sizes': (10, 10, 10, 10), 'activation': 'tanh', 'max_iter': 200, 'solver': 'adam', 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (5, 5, 5), 'activation': 'identity', 'max_iter': 200, 'solver': 'adam', 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (15, 15, 15), 'activation': 'relu', 'max_iter': 300, 'solver': 'sgd', 'learning_rate_init': 0.01},
    {'hidden_layer_sizes': (20, 20, 20), 'activation': 'tanh', 'max_iter': 150, 'solver': 'lbfgs', 'learning_rate_init': 0.005},
    {'hidden_layer_sizes': (25, 25, 25), 'activation': 'logistic', 'max_iter': 250, 'solver': 'adam', 'learning_rate_init': 0.0005},
    {'hidden_layer_sizes': (30, 30, 30), 'activation': 'relu', 'max_iter': 200, 'solver': 'adam', 'learning_rate_init': 0.001},
]

for config in mlp_configs:
    mlp = MLPClassifier(**config, random_state=0)
    mlp.fit(xtrain, ytrain.values.ravel())
    predictions = mlp.predict(xtest)
    matrix = confusion_matrix(ytest, predictions)
    print(matrix)
    target_names = ['Stopped', 'Completed']
    print(classification_report(ytest, predictions, target_names=target_names))
    tn, fp, fn, tp = matrix.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print(f"Config: {config}, Accuracy: {accuracy * 100:.2f}%")