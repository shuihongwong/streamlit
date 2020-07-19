<<<<<<< HEAD
import streamlit as st
import pandas as pd
from sklearn import datasets
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys

sns.set_style('dark')

file = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), 'iris_clf.pkl')
st.write(file)

st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features(mu, max, min):
    sepal_length = st.sidebar.slider('Sepal length', min[0], max[0], mu[0])
    sepal_width = st.sidebar.slider('Sepal width', min[1], max[1], mu[1])
    petal_length = st.sidebar.slider('Petal length', min[2], max[2], mu[2])
    petal_width = st.sidebar.slider('Petal width', min[3], max[3], mu[3])
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

f = lambda x:  "{:.2%}".format(x)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

mu = np.mean(X, axis=0)
max = np.max(X, axis=0)
min = np.min(X, axis=0)

df_p = user_input_features(mu, max, min)

st.subheader('User Input parameters')
st.write(df_p)

df = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])
df['Iris Type'] = df.target.replace(dict(enumerate(iris.target_names)))

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

ax1 = sns.FacetGrid(df,hue="Iris Type",size=3).map(sns.distplot,"sepal length (cm)").add_legend()
st.pyplot()

g = sns.FacetGrid(df, col="Iris Type")
g = g.map(plt.hist, "sepal length (cm)")
st.pyplot()

ax1 = sns.FacetGrid(df,hue="Iris Type",size=3).map(sns.distplot,"sepal width (cm)").add_legend()
st.pyplot()

g = sns.FacetGrid(df, col="Iris Type")
g = g.map(plt.hist, "sepal width (cm)")
st.pyplot()

ax1 = sns.FacetGrid(df,hue="Iris Type",size=3).map(sns.distplot,"petal length (cm)").add_legend()
st.pyplot()

g = sns.FacetGrid(df, col="Iris Type")
g = g.map(plt.hist, "petal length (cm)")
st.pyplot()

ax2 = sns.FacetGrid(df,hue="Iris Type",size=3).map(sns.distplot,"petal width (cm)").add_legend()
st.pyplot()

g = sns.FacetGrid(df, col="Iris Type")
g = g.map(plt.hist, "petal width (cm)")
st.pyplot()

# g = sns.FacetGrid(df, col="Iris Type")
# g = g.map(plt.hist, "sepal length (cm)")
# st.pyplot()

# g = sns.FacetGrid(df, col="Iris Type")
# g = g.map(plt.hist, "sepal width (cm)")
# st.pyplot()

# g = sns.FacetGrid(df, col="Iris Type")
# g = g.map(plt.hist, "petal length (cm)")
# st.pyplot()
# g = sns.FacetGrid(df, col="Iris Type")
# g = g.map(plt.hist, "petal width (cm)")
# st.pyplot()


clf = RandomForestClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
clf.fit(X_train, y_train)

try:
    load_clf = pickle.load(open(file, "rb"))
    st.write('iris_clf.pkl file already exist and the ML algorithm is {}'.format(load_clf))
except (OSError, IOError) as e:
    load_clf = 3
    with open(file, 'wb') as f:
        pickle.dump(clf, f)
    st.write('iris_clf.pkl created')

predictions = clf.predict(X_test)
plt.scatter(y_test, predictions)
plt.xlabel("true label")
plt.ylabel("Predictions")

st.pyplot()

labels = [0, 1, 2]
# labels =['setosa', 'versicolor', 'virginica']
st.subheader("Confusion Matrix for Test Data Set")
cm = confusion_matrix(y_test, predictions, labels)
cm_df = pd.DataFrame(cm, columns = ['setosa', 'versicolor', 'virginica'], index = ['setosa', 'versicolor', 'virginica'])

st.write(cm)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
labels =['setosa', 'versicolor', 'virginica']
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
# plt.show()
st.pyplot()

# load_clf = pickle.load(open('iris_clf.pkl', 'rb'))

st.subheader("Accuracy on Training Set Data")
st.write(f(load_clf.score(X_train, y_train)))

st.subheader("Accuracy on Test Set Data")
st.write(f(load_clf.score(X_test, y_test)))

prediction = load_clf.predict(df_p)
prediction_proba = load_clf.predict_proba(df_p)

st.subheader('Class labels and their corresponding index number')
df = pd.DataFrame(iris.target_names, columns=['Iris_Type'])
st.write(df)

st.subheader('Prediction')
df2 = pd.DataFrame(iris.target_names[prediction], columns=['Predicted_Iris_Type'])

st.write(df2)

st.subheader('Prediction Probability')
# f = lambda x:  "{:.2%}".format(x)
df3 = pd.DataFrame(prediction_proba, columns=iris.target_names)
st.write(df3.applymap(f))

st.bar_chart(df3.T)
=======
import streamlit as st
import pandas as pd
from sklearn import datasets
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('dark')

st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features(mu, max, min):
    sepal_length = st.sidebar.slider('Sepal length', min[0], max[0], mu[0])
    sepal_width = st.sidebar.slider('Sepal width', min[1], max[1], mu[1])
    petal_length = st.sidebar.slider('Petal length', min[2], max[2], mu[2])
    petal_width = st.sidebar.slider('Petal width', min[3], max[3], mu[3])
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

f = lambda x:  "{:.2%}".format(x)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

mu = np.mean(X, axis=0)
max = np.max(X, axis=0)
min = np.min(X, axis=0)

df_p = user_input_features(mu, max, min)

st.subheader('User Input parameters')
st.write(df_p)

df = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])
df['Iris Type'] = df.target.replace(dict(enumerate(iris.target_names)))

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

ax1 = sns.FacetGrid(df,hue="Iris Type",size=3).map(sns.distplot,"sepal length (cm)").add_legend()
st.pyplot()

g = sns.FacetGrid(df, col="Iris Type")
g = g.map(plt.hist, "sepal length (cm)")
st.pyplot()

ax1 = sns.FacetGrid(df,hue="Iris Type",size=3).map(sns.distplot,"sepal width (cm)").add_legend()
st.pyplot()

g = sns.FacetGrid(df, col="Iris Type")
g = g.map(plt.hist, "sepal width (cm)")
st.pyplot()

ax1 = sns.FacetGrid(df,hue="Iris Type",size=3).map(sns.distplot,"petal length (cm)").add_legend()
st.pyplot()

g = sns.FacetGrid(df, col="Iris Type")
g = g.map(plt.hist, "petal length (cm)")
st.pyplot()

ax2 = sns.FacetGrid(df,hue="Iris Type",size=3).map(sns.distplot,"petal width (cm)").add_legend()
st.pyplot()

g = sns.FacetGrid(df, col="Iris Type")
g = g.map(plt.hist, "petal width (cm)")
st.pyplot()

# g = sns.FacetGrid(df, col="Iris Type")
# g = g.map(plt.hist, "sepal length (cm)")
# st.pyplot()

# g = sns.FacetGrid(df, col="Iris Type")
# g = g.map(plt.hist, "sepal width (cm)")
# st.pyplot()

# g = sns.FacetGrid(df, col="Iris Type")
# g = g.map(plt.hist, "petal length (cm)")
# st.pyplot()
# g = sns.FacetGrid(df, col="Iris Type")
# g = g.map(plt.hist, "petal width (cm)")
# st.pyplot()


clf = RandomForestClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
plt.scatter(y_test, predictions)
plt.xlabel("true label")
plt.ylabel("Predictions")

st.pyplot()

labels = [0, 1, 2]
# labels =['setosa', 'versicolor', 'virginica']
st.subheader("Confusion Matrix for Test Data Set")
cm = confusion_matrix(y_test, predictions, labels)
cm_df = pd.DataFrame(cm, columns = ['setosa', 'versicolor', 'virginica'], index = ['setosa', 'versicolor', 'virginica'])

st.write(cm)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
labels =['setosa', 'versicolor', 'virginica']
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
# plt.show()
st.pyplot()

st.subheader("Accuracy on Training Set Data")
st.write(f(clf.score(X_train, y_train)))

st.subheader("Accuracy on Test Set Data")
st.write(f(clf.score(X_test, y_test)))

prediction = clf.predict(df_p)
prediction_proba = clf.predict_proba(df_p)

st.subheader('Class labels and their corresponding index number')
df = pd.DataFrame(iris.target_names, columns=['Iris_Type'])
st.write(df)

st.subheader('Prediction')
df2 = pd.DataFrame(iris.target_names[prediction], columns=['Predicted_Iris_Type'])

st.write(df2)

st.subheader('Prediction Probability')
# f = lambda x:  "{:.2%}".format(x)
df3 = pd.DataFrame(prediction_proba, columns=iris.target_names)
st.write(df3.applymap(f))

st.bar_chart(df3.T)
>>>>>>> 235cef3aef4bf74d729b25d4cd78f377bf72b915
