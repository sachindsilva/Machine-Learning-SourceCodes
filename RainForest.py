# Random Forest

from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# open file with pd.read_csv
df = pd.read_csv("/home/studentdiabetes.csv")
print("Shape of dataset (rows, columns)", df.shape)

# print head of data set
print("\n **************First 5 rows of dataset**********\n", df.head())

X = df.drop('Outcome', axis=1)
y = df['Outcome']

print("\n Features set\n", X.head())
print("\n Target labels\n", y.head())

# implementing train-test-split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# random forest model creation
rfc = RandomForestClassifier(n_estimators=5, max_features=5)
rfc.fit(X_train, y_train)

# predictions
rfc_predict = rfc.predict(X_test)


# Evaluation

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, rfc_predict))
print('\n')
print("=== Classification Report ===")
target_names = ['Diabetes', 'Normal']
print(classification_report(y_test, rfc_predict, target_names=target_names))

index = np.arange(0, len(y_test))
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
plt.scatter(index, y_test, c="red", label='True Value')
plt.scatter(index, rfc_predict, c="blue", label='Predicted Value')

plt.legend()
plt.show()
# Hierarchical Clustering
# Importing the libraries
# Importing the dataset
dataset = pd.read_csv(
    '/home/student/MLLab/ML Lab Datasets/9/Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
# Using the dendrogram to find the optimal number of clusters
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()
# Training the Hierarchical Clustering model on the dataset
hc = AgglomerativeClustering(
    n_clusters=7, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)
# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1],
            s=100, c='red', label='Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1],
            s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1],
            s=100, c='green', label='Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1],
            s=100, c='cyan', label='Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1],
            s=100, c='magenta', label='Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
print("dataset", X)
print("--")
print(X[y_hc == 0, 0])
print("===")
print(X[y_hc == 0, 1])
