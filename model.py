

import pandas as pd
import numpy as np
import pickle
import seaborn as sns
sns.set()

"""IMPORT THE IRIS DATASET FROM THE SKLEARN DATA SET"""

iris_data=pd.read_csv("Iris.csv")

"""DISPLAYING HEAD VALUES OF IRIS_DATA"""

print(iris_data.head())

"""**CHECKING THE NULL OR NaN DATA PRESENT IN THE DATASET**"""

print(iris_data.isnull().sum())

"""PLOTING A PAIR POLT OF EACH FEATURES AGAINST TARGET


**TARGET**

*   0.0-setosa
*   1.0-versicolor
*   2.0-virginica




"""

sns.pairplot(iris_data, hue='Species', size=3 ,palette='Set1')

"""**3Dplotting** relationship between features and the target using ploty.express  """


print(iris_data)


"""PLOTTING THE HEATMAP FOR THE IRIS DATA TO FIND THE REATIONSHIP OF TARGET AND EACH FEATUES
 IN THIS I FOUND THAT THE PETAL WIDTH MOSTLY INFUENCE THE TARGET TO THE NEAREST RELATIONSHIP

ASSIGNING THE TARGET TO THE VARIABLE **"Y"**
"""

Y=iris_data["Species"]
print(Y)

"""ASSIGNING ALL THE FEATURES TO THE VARIABLE **"X" **BY DROPING THE TARGET FROM THE IRIS DATA"""

X_old=iris_data.drop("Species", axis=1)

X=X_old.drop("Id", axis=1)
print(X)

"""SPLITTING THE TEST AND TRAIN DATA OF IRIS DATA SET USING TEST_TRAIN_SPLIT MODULE OF SKLEARN.MODEL_SELECTION"""

from sklearn.model_selection import train_test_split

"""SPLITING THE TEST AND TRAIN DATA OF IRIS DATA ASIGN TEST **DATA 20% AND TRAIN DATA 80%**"""

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,random_state=1)

"""BY USING STANDARDSCALER FROM SKLEARN.PREPROCESSING TO **NORMALIZE** THE DATAS FROM THE DATA SET FOR BOTH TEST AND TRAIN DATAS"""

from sklearn.preprocessing import MinMaxScaler

std_scale = MinMaxScaler()
std_scale.fit_transform(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test)

"""NORMALIZED TRAIN_X VALUE"""

print(X_train_std)

"""NORMALIZED TEST_X VALUE"""

print(X_test_std)

"""VISUVALISING THE NON STANDARD DATA(**NOT NORMALIZED**) USING HEAT MAP """

sns.heatmap(X_train)

"""VISUVALISING THE STANDARD DATA(**NORMALIZED**) USING HEAT MAP """

sns.heatmap(X_train_std)

"""IMPORT THE KNeighborsClassifier FROM THE sklearn.neighbors FOR FIND TO PERFORM THE TRAING USING KNN ALGORITHM AND ASSIGN NEIGHBOURS=5

"""

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train,y_train)

y_predict=knn.predict(X_test)
print(y_predict)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,y_predict))
print(confusion_matrix(y_test,y_predict))

cf_matrix = confusion_matrix(y_test, y_predict)
sns.heatmap(cf_matrix, annot=True)

sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,
            fmt='.1%', cmap='Purples')

"""IMPORTING ACCURACY SCORE MODULE FROM SKLEARN.METRICS TO GET THE ACCURACY OF TEST AND PREDICTED DATASET"""

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test, y_predict)
print("THE ACCURACY OF THE KNN MODEL FOR THIS IRIS DATA SET  : ",(accuracy*100),"%")



model=knn


pickle.dump(model,open('model.pkl','wb'))
pickle.dump(std_scale,open('scale.pkl','wb'))