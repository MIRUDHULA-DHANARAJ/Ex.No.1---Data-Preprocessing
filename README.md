# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train

## PROGRAM:
~~~
import pandas as pd

df = pd.read_csv("Churn_Modelling.csv")

df.head()
df.info()

x = df.iloc[:,:-1].values
y= df.iloc[:,1].values
x
y

df.describe()


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df1 = df.copy()

df1["Geography"] = le.fit_transform(df1["Geography"])
df1["Gender"] = le.fit_transform(df1["Gender"])


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df1[["CreditScore","Geography","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]] = pd.DataFrame(scaler.fit_transform(df1[["CreditScore","Geography","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]]))



df1.describe()


X = df1[["CreditScore","Geography","Gender","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]].values
print(X)

y = df1.iloc[:,-1].values
print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train)

print("Size of X_train: ",len(X_train))

print(X_test)
print("Size of X_test: ",len(X_test))

~~~

## OUTPUT:

## HEAD:

![image](https://user-images.githubusercontent.com/94828147/229674087-b333ecdd-8431-4aa8-b1f1-bd1b5d2751d6.png)

## INFO:

![image](https://user-images.githubusercontent.com/94828147/229674127-dd12beb4-f7f2-4557-8995-0c95471e137a.png)

![image](https://user-images.githubusercontent.com/94828147/229674197-bef16c23-3655-444e-ae8e-e1e3715ffcdd.png)

![image](https://user-images.githubusercontent.com/94828147/229674217-ec5918c8-4855-4beb-ae30-0168c7b6c18f.png)

## BEFORE NORMALIZING:

![image](https://user-images.githubusercontent.com/94828147/229674322-a061c5f6-b2a2-47c3-87ff-48c4e2a995b9.png)

## AFTER NORMALIZING:

![image](https://user-images.githubusercontent.com/94828147/229674379-82bbf5d3-fd95-4649-9c32-95a42af7120e.png)

## SIZE OF XTRAIN:

![image](https://user-images.githubusercontent.com/94828147/229674494-0a870489-737d-418a-8c2e-6a17fdb0f2ae.png)

## SIZE OF YTRAIN:

![image](https://user-images.githubusercontent.com/94828147/229674722-3bbbd97e-bb91-4d9c-ab6a-360848cc2324.png)



## RESULT
Data preprocessing is performed in the given dataset.
