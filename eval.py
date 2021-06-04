import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate
from sklearn.impute import KNNImputer
from utils import train_and_test
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer


def transform_one_hot(data, ind, labels):
    n, d = data.shape
    one_hot = np.zeros((n, len(labels)))
    for i in range(len(data[:,ind])):
        for j in range(len(labels)):
            if data[i, ind] == labels[j]:
                one_hot[i,j] = 1
                break

    new_data = np.concatenate((data[:,0:ind].reshape((n,ind)), one_hot, 
                                data[:,ind+1:]), axis = 1)
    return new_data

def process(dat):
    #  new_data = dat[['ID', 'Sex',
                    #  'Age', 'Height', 'location',
                    #  'Weight',
                    #  'mean.Temperature_60',
                    #  'grad.Temperature_60',
                    #  'sd.Temperature_60',
                    #  'mean.Temperature_480',
                    #  'grad.Temperature_480',
                    #  'sd.Temperature_480',
                    #  'mean.Humidity_60',
                    #  'grad.Humidity_60',
                    #  'sd.Humidity_60',
                    #  'mean.Humidity_480',
                    #  'grad.Humidity_480',
                    #  'sd.Humidity_480',
                    #  'mean.Solar_60',
                    #  'grad.Solar_60',
                    #  'sd.Solar_60',
                    #  'mean.Solar_480',
                    #  'grad.Solar_480',
                    #  'sd.Solar_480',
                    #  'mean.hr_5',
                    #  'grad.hr_5',
                    #  'sd.hr_5',
                    #  'mean.hr_15',
                    #  'grad.hr_15',
                    #  'sd.hr_15',
                    #  'mean.hr_60',
                    #  'grad.hr_60',
                    #  'sd.hr_60',
                    #  'mean.WristT_5',
                    #  'grad.WristT_5',
                    #  'sd.WristT_5',
                    #  'mean.WristT_15',
                    #  'grad.WristT_15',
                    #  'sd.WristT_15',
                    #  'mean.WristT_60',
                    #  'grad.WristT_60',
                    #  'sd.WristT_60',
                    #  'mean.AnkleT_5',
                    #  'grad.AnkleT_5',
                    #  'sd.AnkleT_5',
                    #  'mean.AnkleT_15',
                    #  'grad.AnkleT_15',
                    #  'sd.AnkleT_15',
                    #  'mean.AnkleT_60',
                    #  'grad.AnkleT_60',
                    #  'sd.AnkleT_60']]

    new_data = dat.loc[: , dat.columns!= 'therm_pref']
    new_data = new_data.loc[: , new_data.columns != 'Vote_time']
    new_data = new_data.loc[: , new_data.columns != 'therm_sens']

    columns = new_data.columns
    new_data = np.array(new_data)
    new_data[new_data == 'Male'] = 0
    new_data[new_data == 'Female'] = 1

    #  new_data = transform_one_hot(new_data, 4, [-1,1])
    #  new_data = transform_one_hot(new_data, 1, ['Male', 'Female'])

    new_data = new_data.astype(float)
    n, _ = new_data.shape

    # Indices in new data that does not
    # contain nan in its row. If false
    # then new data at the same index contains nan
    # in its row

    not_nan_indices = np.ones(n, dtype=bool)

    for i in range(n):
        for j in new_data[i]:
            if j!=j:
                not_nan_indices[i] = False
                break

    print('KNN Imputer')

    imputer = KNNImputer(n_neighbors=10)
    new_data = imputer.fit_transform(new_data)

    new_data = pandas.DataFrame(data=new_data, columns=columns)
    new_data.to_csv('./data/imputed_wearable.csv')

    #  new_data = pandas.DataFrame(new_data)
    #  df = new_data.interpolate(method='spline', order=5)
    #  new_data = np.array(df.ffill().bfill())

    y = dat['therm_pref']
    y += 1
    X = np.array(new_data, dtype=float)
    y = np.array(y, dtype=int)
    #  Xtrain = X[not_nan_indices == False]
    #  ytrain = y[not_nan_indices == False]
    #  Xtest = X[not_nan_indices == True]
    #  ytest = y[not_nan_indices == True]
    #  return Xtrain, ytrain, Xtest, ytest
    return X, y, not_nan_indices


dat = pandas.read_csv('./data/wearable.csv')
X, y, not_nan_indices = process(dat)
counts = np.bincount(y)
plt.bar(("Too Hot", "Comfortable", "Too Cold"), counts)
plt.show()
colors = []
gen = []
n, _ = X.shape
# Red is generated and blue os original
for i in range(n):
    col = 'red'
    g = 'Imputed Data'
    if not_nan_indices[i] == True:
        col = 'blue'
        g = 'Non Imputed Data'
    colors.append(col)

pca = PCA(n_components=2)
transf = pca.fit_transform(X)
feat1 = transf[:, 0]
feat2 = transf[:, 1]
colors = np.array(colors)


scatter_x = feat1[y==0]
scatter_y = feat2[y==0]
group = not_nan_indices[y==0]
fig, ax = plt.subplots()
ix = np.where(group == True)
ax.scatter(scatter_x[ix], scatter_y[ix], c = 'blue', label = 'Orignal')
ix = np.where(group == False)
ax.scatter(scatter_x[ix], scatter_y[ix], c = 'red', label = 'Imputed')
ax.legend()
plt.title('Median Imputation for class 1')
plt.show()

scatter_x = feat1[y==1]
scatter_y = feat2[y==1]
group = not_nan_indices[y==1]
fig, ax = plt.subplots()
ix = np.where(group == True)
ax.scatter(scatter_x[ix], scatter_y[ix], c = 'blue', label = 'Orignal')
ix = np.where(group == False)
ax.scatter(scatter_x[ix], scatter_y[ix], c = 'red', label = 'Imputed')
ax.legend()
plt.title('Median Imputation for class 2')
plt.show()


scatter_x = feat1[y==2]
scatter_y = feat2[y==2]
group = not_nan_indices[y==2]
fig, ax = plt.subplots()
ix = np.where(group == True)
ax.scatter(scatter_x[ix], scatter_y[ix], c = 'blue', label = 'Orignal')
ix = np.where(group == False)
ax.scatter(scatter_x[ix], scatter_y[ix], c = 'red', label = 'Imputed')
ax.legend()
plt.title('Median Imputation for class 3')
plt.show()


#  model = RandomForestClassifier()
#  results = train_and_test(model, Xtrain, ytrain, Xtest, 
                        #  ytest)
#  print('Random Forest')
#  print('Training Accuracy: '+str(np.mean(results['train'])))
#  print('Testing Accuracy: '+str(np.mean(results['test'])))

#  model = SVC()
#  results = train_and_test(model, Xtrain, ytrain, Xtest, 
                        #  ytest)
#  print('SVC')
#  print('Training Accuracy: '+str(np.mean(results['train'])))
#  print('Testing Accuracy: '+str(np.mean(results['test'])))

#  model = KNeighborsClassifier(n_neighbors=10)
#  results = train_and_test(model, Xtrain, ytrain, Xtest, 
                        #  ytest)
#  print('KNN')
#  print('Training Accuracy: '+str(np.mean(results['train'])))
#  print('Testing Accuracy: '+str(np.mean(results['test'])))

#  model = GaussianNB()
#  results = train_and_test(model, Xtrain, ytrain, Xtest, 
                        #  ytest)
#  print('Naive Bayes')
#  print('Training Accuracy: '+str(np.mean(results['train'])))
#  print('Testing Accuracy: '+str(np.mean(results['test'])))

