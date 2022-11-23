#!/usr/bin/env python
# coding: utf-8

# - [x] Independently implement one of the classification methods, with the ability to adjust hyperparameters.

# In[42]:


import numpy as np

class LogRegression:
    
    def __init__(self, learn_rate=0.001, num_iters=1000):
        self.learn_rate = learn_rate
        self.num_iters = num_iters
        self.w = None
        self.b = None
    
    def sigmoid(self, x):
        return(1 / (1 + np.exp(-x)))
    
    def fit(self, X, y):
        n, m = X.shape
        self.w = np.zeros(m)
        self.b = 0
        
        for i in range(self.num_iters):
            f = np.dot(X, self.w) + self.b
            y_pred = self.sigmoid(f)
            d_w = (1 / n) * np.dot(X.T, (y_pred - y))
            d_b = (1 / n) * np.sum(y_pred - y)
            self.w = self.w - self.learn_rate * d_w
            self.b = self.b - self.learn_rate * d_b
            
    def predict(self, X):
        f = np.dot(X, self.w) + self.b
        y_pred = self.sigmoid(f)
        y_pred_class = [1 if j > 0.5 else 0 for j in y_pred]
        return y_pred_class        


# - [x] Take data to predict heart disease [here](https://github.com/rustam-azimov/ml-course/tree/main/data/heart_disease) (target feature for prediction --- **target** ). A demo notebook with the analysis of this data can be found [here](https://github.com/rustam-azimov/ml-course/blob/main/practice/practice07_knn_nb/practice07_part02_classification_heart_disease_demo.ipynb)

# In[134]:


import pandas as pd

df = pd.read_csv('data/heart.csv')

df


# - [x] Read data, perform initial data analysis, if necessary, clean the data (**Data Cleaning**).

# ### Data explanation
# 
# Возьмём данные заболеваний сердца у пациентов, которые можно скачать [тут](https://www.kaggle.com/code/ahmedadell30/heart-disease-prediction-with-ann-models/data).
# 
# Описание датасета: <br>
# 
# * age - age in years <br>
# * sex - (1 = male; 0 = female) <br>
# * cp - chest pain type <br>
# * trestbps - resting blood pressure (in mm Hg on admission to the hospital) <br>
# * chol - serum cholestoral in mg/dl <br>
# * fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false) <br>
# * restecg - resting electrocardiographic results <br>
# * thalach - maximum heart rate achieved <br>
# * exang - exercise induced angina (1 = yes; 0 = no) <br>
# * oldpeak - ST depression induced by exercise relative to rest <br>
# * slope - the slope of the peak exercise ST segment <br>
# * ca - number of major vessels (0-3) colored by flourosopy <br>
# * thal - 3 = normal; 6 = fixed defect; 7 = reversable defect <br>
# * target - have disease or not (1=yes, 0=no)
# 
# Необходимо решить задачу классификации и научиться предсказывать целовой признак **target** имеет ли пациент заболевание сердца.

# In[137]:


df.info()


# In[44]:


import seaborn as sns

sns.pairplot(df)


# In[143]:


correlation = df.corr()

mask = np.zeros_like(correlation, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(20, 20))

cmap = sns.diverging_palette(180, 20, as_cmap=True)
sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=1, vmin =-1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.show()


# In[46]:


df = df.dropna()

df


# In[138]:


pd.DataFrame(df.apply(lambda col: len(col.unique())),columns=["Unique Values Count"])


# In[141]:


df.describe().T


# - [x] Perform exploratory analysis (**EDA**), use visualization, draw conclusions that may be useful in further solving the classification problem.

# In[51]:


df["age"].plot.hist()


# In[53]:


df["sex"].plot.hist()


# In[55]:


df["cp"].plot.hist() # This feature had higher positive correlation with the target


# In[64]:


df["thalach"].unique() # This feature had higher positive correlation with the target


# In[67]:


df["exang"].plot.hist() # This feature had higher negative correlation with the target


# In[69]:


df["oldpeak"].plot.hist() # This feature had higher negative correlation with the target


# In[71]:


df["slope"].plot.hist() # This feature had higher positive correlation with the target


# In[73]:


df["ca"].plot.hist() # This feature had higher negative correlation with the target


# In[77]:


df["target"].plot.hist()


# In[78]:


df.groupby("target").mean()


# In[80]:


import matplotlib.pyplot as plt

pd.crosstab(df.age,df.target).plot(kind="bar", figsize=(20, 6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[81]:


pd.crosstab(df.sex,df.target).plot(kind="bar", figsize=(15, 6))
plt.title('Heart Disease Frequency for Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["No Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()


# In[82]:


pd.crosstab(df.cp,df.target).plot(kind="bar", figsize=(20, 6))
plt.title('Heart Disease Frequency for Chest pain types')
plt.xlabel('Chest pain type')
plt.ylabel('Frequency')
plt.show()


# In[83]:


pd.crosstab(df.fbs,df.target).plot(kind="bar", figsize=(15, 6))
plt.title('Heart Disease Frequency for FBS levels')
plt.xlabel('FBS - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)')
plt.xticks(rotation=0)
plt.legend(["No Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()


# In[84]:


plt.scatter(x=df.trestbps[df.target==1], y=df.thalach[(df.target==1)])
plt.scatter(x=df.trestbps[df.target==0], y=df.thalach[(df.target==0)])
plt.legend(["Disease", "No Disease"])
plt.xlabel("Resting blood pressure")
plt.ylabel("Maximum Heart Rate")
plt.show()


# In[85]:


plt.scatter(x=df.age[df.target==1], y=df.chol[(df.target==1)])
plt.scatter(x=df.age[df.target==0], y=df.chol[(df.target==0)])
plt.legend(["Disease", "No Disease"])
plt.xlabel("Age")
plt.ylabel("Serum cholesterol level")
plt.show()


# In[86]:


plt.scatter(x=df.oldpeak[df.target==1], y=df.chol[(df.target==1)])
plt.scatter(x=df.oldpeak[df.target==0], y=df.chol[(df.target==0)])
plt.legend(["Disease", "No Disease"])
plt.xlabel("ST depression induced by exercise relative to test")
plt.ylabel("Serum cholesterol level")
plt.show()


# - [x] If necessary, perform useful data transformations (for example, transform categorical features into quantitative ones), remove unnecessary features, create new ones (**Feature Engineering**).
# 

# In[88]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X, y = df.drop(columns=["target"]), df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# In[89]:


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred)/len(y_true)
    return accuracy


# In[90]:


log_reg = LogRegression(learn_rate=0.0001, num_iters=1000)
log_reg.fit(X_train, y_train)

pred = log_reg.predict(X_test)

print("Accuracy: ", accuracy(y_test, pred))


# - [x] Using **hyperparameter matching**, **cross-validation** and, if necessary, **data scaling**, achieve the best prediction quality from your implementation on a preselected test set.
# 

# In[91]:


log_reg = LogRegression(learn_rate=0.00005, num_iters=1000)
log_reg.fit(X_train, y_train)

pred = log_reg.predict(X_test)

print("Accuracy: ", accuracy(y_test, pred))


# In[92]:


log_reg = LogRegression(learn_rate=0.00005, num_iters=1500)
log_reg.fit(X_train, y_train)

pred = log_reg.predict(X_test)

print("Accuracy: ", accuracy(y_test, pred))


# In[93]:


log_reg = LogRegression(learn_rate=0.00005, num_iters=1500)
log_reg.fit(X_train_scaled, y_train)

pred = log_reg.predict(X_test_scaled)

print("Accuracy: ", accuracy(y_test, pred))


# - [x] Repeat the previous point for library implementations (e.g. from **sklearn**) of all passed classification methods (**logistic regression, svm, knn, naive bayes, decision tree**).
# 

# In[94]:


from sklearn.linear_model import LogisticRegression
accuracies = {}

lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)
acc = lr.score(X_test_scaled, y_test)

accuracies['Logistic Regression'] = acc
print("Test Accuracy {:.4f}".format(acc))


# In[95]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)  # n_neighbors means k
knn.fit(X_train_scaled, y_train)
prediction = knn.predict(X_test_scaled)

print("{} NN Score: {:.4f}".format(3, knn.score(X_test_scaled, y_test)))


# In[96]:


scoreList = []
for i in range(1, 20, 2):
    knn2 = KNeighborsClassifier(n_neighbors = i)
    knn2.fit(X_train_scaled, y_train)
    scoreList.append(knn2.score(X_test_scaled, y_test))
    
plt.plot(range(1, 20, 2), scoreList)
plt.xticks(np.arange(1, 20, 2))
plt.xlabel("K value")
plt.ylabel("Score")
plt.show()

acc = max(scoreList)
accuracies['KNN'] = acc
print("Maximum KNN Score is {:.4f}".format(acc))


# In[97]:


from sklearn.svm import SVC

svm = SVC(random_state = 12)
svm.fit(X_train_scaled, y_train)

acc = svm.score(X_test_scaled, y_test)
accuracies['SVM'] = acc
print("Test Accuracy of SVM Algorithm: {:.4f}".format(acc))


# In[98]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)

acc = nb.score(X_test_scaled, y_test)
accuracies['Naive Bayes'] = acc
print("Accuracy of Naive Bayes: {:.4f}".format(acc))


# In[99]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

acc = dtc.score(X_test, y_test)
accuracies['Decision Tree'] = acc
print("Decision Tree Test Accuracy {:.4f}".format(acc))


# In[100]:


from sklearn import tree

plt.figure(figsize=(32, 32))
tree.plot_tree(dtc, feature_names=df.columns,fontsize=10);


# - [x] Compare all trained models, build their **confusion matrices**. Draw conclusions about the models obtained in the framework of solving the classification problem on the selected data.
# 

# In[101]:


accuracies["Logistic Regression from scratch"] = accuracy(y_test, pred)


# In[102]:


colors = ["purple", "green", "orange", "magenta", "blue", "black"]

sns.set_style("whitegrid")
plt.figure(figsize=(16, 5))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.ylabel("Accuracy")
plt.xlabel("Algorithms")
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors)
plt.show()


# In[103]:


# Predicted values
y_head_lr = lr.predict(X_test_scaled)
knn3 = KNeighborsClassifier(n_neighbors = 1)
knn3.fit(X_train_scaled, y_train)
y_head_knn = knn3.predict(X_test_scaled)
y_head_svm = svm.predict(X_test_scaled)
y_head_nb = nb.predict(X_test_scaled)
y_head_dtc = dtc.predict(X_test)
y_head_lrs = log_reg.predict(X_test_scaled)


# In[104]:


from sklearn.metrics import confusion_matrix

cm_lr = confusion_matrix(y_test, y_head_lr)
cm_knn = confusion_matrix(y_test, y_head_knn)
cm_svm = confusion_matrix(y_test, y_head_svm)
cm_nb = confusion_matrix(y_test, y_head_nb)
cm_dtc = confusion_matrix(y_test, y_head_dtc)
cm_lrs = confusion_matrix(y_test, y_head_lrs)


# In[113]:


plt.figure(figsize=(24, 12))

plt.suptitle("Confusion Matrices", fontsize=24)
plt.subplots_adjust(wspace = 0.4, hspace= 0.4)

plt.subplot(2,3,1)
plt.title("Logistic Regression Confusion Matrix")
sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,2)
plt.title("K Nearest Neighbors Confusion Matrix")
sns.heatmap(cm_knn,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,3)
plt.title("Support Vector Machine Confusion Matrix")
sns.heatmap(cm_svm,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,4)
plt.title("Naive Bayes Confusion Matrix")
sns.heatmap(cm_nb,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,5)
plt.title("Decision Tree Classifier Confusion Matrix")
sns.heatmap(cm_dtc,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,6)
plt.title("Logistic Regression from Scratch Confusion Matrix")
sns.heatmap(cm_lrs,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.show()


# - [x] (**+2 points**) Implement one more of the classification methods and add it to the comparison.

# In[106]:


import numpy as np

class NaiveBayes:
    
    def fit(self, X, y):
        n, m = X.shape
        self.classes = np.unique(y)
        num_classes = len(self.classes)

        self.mean = np.zeros((num_classes, m), dtype=np.float64)
        self.var = np.zeros((num_classes, m), dtype=np.float64)
        self.priors = np.zeros(num_classes, dtype=np.float64)

        for c in self.classes:
            X_c = X[c==y]
            self.mean[c,:]=X_c.mean(axis=0)
            self.var[c,:]=X_c.var(axis=0)
            self.priors[c] = X_c.shape[0] / float(n)

    def pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-(x-mean)**2/(2 * var))
        denominator = np.sqrt(2*np.pi*var)
        return(numerator/denominator)            
            
    def pred(self, x):
        posteriors = []
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            class_conditional = np.sum(np.log(self.pdf(idx,x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        return(self.classes[np.argmax(posteriors)])
    
    def predict(self,X):
        y_pred = [self.pred(x) for x in X]
        return(y_pred)


# In[109]:


import matplotlib.pyplot as plt

nbs = NaiveBayes()
nbs.fit(X_train_scaled, y_train)
pred_nbs = nb.predict(X_test_scaled)

print("Naive Bayes classification accuracy ", accuracy(y_test, pred_nbs))


# In[110]:


accuracies["Naive Bayes from scratch"] = accuracy(y_test, pred_nbs)


# In[111]:


colors = ["purple", "green", "orange", "magenta", "blue", "black", "gray"]

sns.set_style("whitegrid")
plt.figure(figsize=(16, 5))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.ylabel("Accuracy")
plt.xlabel("Algorithms")
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors)
plt.show()


# In[112]:


y_head_nbs = nbs.predict(X_test_scaled)
cm_nbs = confusion_matrix(y_test, y_head_nbs)


# In[116]:


plt.figure(figsize=(24, 15))

plt.suptitle("Confusion Matrices", fontsize=24)
plt.subplots_adjust(wspace = 0.4, hspace= 0.4)

plt.subplot(3,3,1)
plt.title("Logistic Regression Confusion Matrix")
sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(3,3,2)
plt.title("K Nearest Neighbors Confusion Matrix")
sns.heatmap(cm_knn,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(3,3,3)
plt.title("Support Vector Machine Confusion Matrix")
sns.heatmap(cm_svm,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(3,3,4)
plt.title("Naive Bayes Confusion Matrix")
sns.heatmap(cm_nb,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(3,3,5)
plt.title("Decision Tree Classifier Confusion Matrix")
sns.heatmap(cm_dtc,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(3,3,6)
plt.title("Logistic Regression from Scratch Confusion Matrix")
sns.heatmap(cm_lrs,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(3,3,7)
plt.title("Naive Bayes from Scratch Confusion Matrix")
sns.heatmap(cm_nbs,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.show()


# - [x] (**+2 points**) Find data on which it will be interesting to solve the classification problem. Repeat all the tasks on the new data.
# 

# In[162]:


df_2 = pd.read_csv('data/wbcd.csv')

df_2


# Attribute Information:
# 
# => Diagnosis (M = malignant, B = benign))
# 
# Ten real-valued features are computed for each cell nucleus:
# 
# - radius (mean of distances from center to points on the perimeter)
# - texture (standard deviation of gray-scale values)
# - perimeter
# - area
# - smoothness (local variation in radius lengths)
# - compactness (perimeter^2 / area - 1.0)
# - concavity (severity of concave portions of the contour)
# - concave points (number of concave portions of the contour)
# - symmetry
# - fractal dimension ("coastline approximation" - 1)

# In[163]:


df_2.drop(["id", "Unnamed: 32"], axis=1, inplace=True)

df_2


# In[164]:


df_2["diagnosis"] = df_2["diagnosis"].map({"M":1, "B":0})

df_2


# In[165]:


df = df.dropna()

df


# In[153]:


# sns.pairplot(df_2)


# In[166]:


correlation = df_2.corr()

mask = np.zeros_like(correlation, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(20, 20))

cmap = sns.diverging_palette(180, 20, as_cmap=True)
sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=1, vmin =-1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.show()


# In[167]:


pd.DataFrame(df_2.apply(lambda col: len(col.unique())),columns=["Unique Values Count"])


# In[168]:


df_2.describe().T


# In[169]:


df["diagnosis"].plot.hist()


# In[170]:


df.groupby("diagnosis").mean()


# In[171]:


pd.crosstab(df_2.area_mean,df_2.diagnosis).plot(kind="bar", figsize=(20, 6))
plt.title('Malignant tumor frequency for Averaged area')
plt.xlabel('Averaged area')
plt.ylabel('Frequency')
plt.show()

# Since the data is so granular, these plots are not suitable


# In[173]:


plt.scatter(x=df_2.area_mean[df_2.diagnosis==1], y=df_2.texture_mean[(df_2.diagnosis==1)])
plt.scatter(x=df_2.area_mean[df_2.diagnosis==0], y=df_2.texture_mean[(df_2.diagnosis==0)])
plt.legend(["Malignant", "Benign"])
plt.xlabel("Averaged area")
plt.ylabel("Averaged texture")
plt.show()

# These are good attributes


# In[174]:


plt.scatter(x=df_2.smoothness_mean[df_2.diagnosis==1], y=df_2.symmetry_mean[(df_2.diagnosis==1)])
plt.scatter(x=df_2.smoothness_mean[df_2.diagnosis==0], y=df_2.symmetry_mean[(df_2.diagnosis==0)])
plt.legend(["Malignant", "Benign"])
plt.xlabel("Averaged smoothness")
plt.ylabel("Averaged symmetry")
plt.show()

# These are not good attributes


# In[175]:


plt.scatter(x=df_2.concavity_mean[df_2.diagnosis==1], y=df_2.concavity_worst[(df_2.diagnosis==1)])
plt.scatter(x=df_2.concavity_mean[df_2.diagnosis==0], y=df_2.concavity_worst[(df_2.diagnosis==0)])
plt.legend(["Malignant", "Benign"])
plt.xlabel("Averaged concavity")
plt.ylabel("Worst concavity")
plt.show()

# Thought of using only averaged values, decided to use all


# In[176]:


plt.scatter(x=df_2.fractal_dimension_se[df_2.diagnosis==1], y=df_2.compactness_se[(df_2.diagnosis==1)])
plt.scatter(x=df_2.fractal_dimension_se[df_2.diagnosis==0], y=df_2.compactness_se[(df_2.diagnosis==0)])
plt.legend(["Malignant", "Benign"])
plt.xlabel("Fractal dimension")
plt.ylabel("Compactness")
plt.show()

# Not that good


# In[182]:


from sklearn.preprocessing import StandardScaler

X, y = df_2.drop(columns=["diagnosis"]), df_2["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
# Did not use MinMaxScaler here

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# In[187]:


log_reg = LogRegression(learn_rate=0.0001, num_iters=1500)
log_reg.fit(X_train_scaled, y_train)

pred = log_reg.predict(X_test_scaled)

print("Accuracy: ", accuracy(y_test, pred))


# In[201]:


accuracies["Logistic Regression from scratch"] = accuracy(y_test, pred)


# In[188]:


accuracies = {}

lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)
acc = lr.score(X_test_scaled, y_test)

accuracies['Logistic Regression'] = acc
print("Test Accuracy {:.4f}".format(acc))


# In[189]:


knn = KNeighborsClassifier(n_neighbors = 3)  # n_neighbors means k
knn.fit(X_train_scaled, y_train)
prediction = knn.predict(X_test_scaled)

print("{} NN Score: {:.4f}".format(3, knn.score(X_test_scaled, y_test)))


# In[190]:


scoreList = []
for i in range(1, 20, 2):
    knn2 = KNeighborsClassifier(n_neighbors = i)
    knn2.fit(X_train_scaled, y_train)
    scoreList.append(knn2.score(X_test_scaled, y_test))
    
plt.plot(range(1, 20, 2), scoreList)
plt.xticks(np.arange(1, 20, 2))
plt.xlabel("K value")
plt.ylabel("Score")
plt.show()

acc = max(scoreList)
accuracies['KNN'] = acc
print("Maximum KNN Score is {:.4f}".format(acc))


# In[191]:


svm = SVC(random_state = 12)
svm.fit(X_train_scaled, y_train)

acc = svm.score(X_test_scaled, y_test)
accuracies['SVM'] = acc
print("Test Accuracy of SVM Algorithm: {:.4f}".format(acc))


# In[192]:


nb = GaussianNB()
nb.fit(X_train_scaled, y_train)

acc = nb.score(X_test_scaled, y_test)
accuracies['Naive Bayes'] = acc
print("Accuracy of Naive Bayes: {:.4f}".format(acc))


# In[193]:


dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

acc = dtc.score(X_test, y_test)
accuracies['Decision Tree'] = acc
print("Decision Tree Test Accuracy {:.4f}".format(acc))


# In[194]:


plt.figure(figsize=(32, 32))
tree.plot_tree(dtc, feature_names=df_2.columns,fontsize=10);


# In[198]:


nbs = NaiveBayes()
nbs.fit(X_train_scaled, y_train)
pred_nbs = nb.predict(X_test_scaled)

print("Naive Bayes classification accuracy ", accuracy(y_test, pred_nbs))


# In[199]:


accuracies["Naive Bayes from scratch"] = accuracy(y_test, pred_nbs)


# In[202]:


colors = ["purple", "green", "orange", "magenta", "blue", "black", "gray"]

sns.set_style("whitegrid")
plt.figure(figsize=(16, 5))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.ylabel("Accuracy")
plt.xlabel("Algorithms")
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors)
plt.show()


# In[205]:


y_head_lr = lr.predict(X_test_scaled)
knn3 = KNeighborsClassifier(n_neighbors = 1)
knn3.fit(X_train_scaled, y_train)
y_head_knn = knn3.predict(X_test_scaled)
y_head_svm = svm.predict(X_test_scaled)
y_head_nb = nb.predict(X_test_scaled)
y_head_dtc = dtc.predict(X_test)
y_head_lrs = log_reg.predict(X_test_scaled)
y_head_nbs = nbs.predict(X_test_scaled)


# In[206]:


cm_lr = confusion_matrix(y_test, y_head_lr)
cm_knn = confusion_matrix(y_test, y_head_knn)
cm_svm = confusion_matrix(y_test, y_head_svm)
cm_nb = confusion_matrix(y_test, y_head_nb)
cm_dtc = confusion_matrix(y_test, y_head_dtc)
cm_lrs = confusion_matrix(y_test, y_head_lrs)
cm_nbs = confusion_matrix(y_test, y_head_nbs)


# In[207]:


plt.figure(figsize=(24, 15))

plt.suptitle("Confusion Matrices", fontsize=24)
plt.subplots_adjust(wspace = 0.4, hspace= 0.4)

plt.subplot(3,3,1)
plt.title("Logistic Regression Confusion Matrix")
sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(3,3,2)
plt.title("K Nearest Neighbors Confusion Matrix")
sns.heatmap(cm_knn,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(3,3,3)
plt.title("Support Vector Machine Confusion Matrix")
sns.heatmap(cm_svm,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(3,3,4)
plt.title("Naive Bayes Confusion Matrix")
sns.heatmap(cm_nb,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(3,3,5)
plt.title("Decision Tree Classifier Confusion Matrix")
sns.heatmap(cm_dtc,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(3,3,6)
plt.title("Logistic Regression from Scratch Confusion Matrix")
sns.heatmap(cm_lrs,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(3,3,7)
plt.title("Naive Bayes from Scratch Confusion Matrix")
sns.heatmap(cm_nbs,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.show()

