import pandas as pd

from sklearn.model_selection import train_test_split import numpy as np

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder

data=pd.read_csv("/home/student/4nm20is120/DecisionTree/tennis.csv")

print("--Tennis Dataset--",data)



col=data.columns

print("--The Columns are--\n")

print(col)

feature=col[1:5]

print(feature)



for label in col:

data[label]=LabelEncoder().fit_transform(data[label])

data1=data.drop(['play'],axis=1)

# Note : axis=1 indicates 'columns'

print('---The data1 dataset---\n')

print(data1)



target=data['play']

print('---Target values---\n\n')

print(target)



x_train,x_test,y_train,y_test=train_test_split(data1,target,test_size=0.2)

print("--	The Training Set--\n",y_train)	
print("--	Training Set Label--\n",y_test)
print("--	Testing Target--\n",y_test)	



id3=DecisionTreeClassifier()

id3=id3.fit(x_train,y_train)

y_predict=id3.predict(x_test)

print("--Actual Output--\n\n")

print("--Actual Output--\n\n")
 
print(y_test)

print("--Predicted Output--\n\n")

print(y_predict)


Output:
import pandas as pd

from sklearn.model_selection import train_test_split import numpy as np

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder

data=pd.read_csv("/home/student/4nm20is120/DecisionTree/tennis.csv")

print("--Tennis Dataset--",data)



col=data.columns

print("--The Columns are--\n")

print(col)

feature=col[1:5]

print(feature)



for label in col:

data[label]=LabelEncoder().fit_transform(data[label])

data1=data.drop(['play'],axis=1)

# Note : axis=1 indicates 'columns'

print('---The data1 dataset---\n')

print(data1)



target=data['play']

print('---Target values---\n\n')

print(target)



x_train,x_test,y_train,y_test=train_test_split(data1,target,test_size=0.2)

print("--	The Training Set--\n",y_train)	
print("--	Training Set Label--\n",y_test)
print("--	Testing Target--\n",y_test)	
 
id3=DecisionTreeClassifier()

id3=id3.fit(x_train,y_train)

y_predict=id3.predict(x_test)

print("--Actual Output--\n\n")

print("--Actual Output--\n\n")

print(y_test)

print("--Predicted Output--\n\n")

print(y_predict)
