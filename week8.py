import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import time

stemmer = SnowballStemmer('english')
stemming = input("Do you want to use stemming? (y/n): ")
#df_attributes = pd.read_csv('attributes.csv/attributes.csv')
df_train = pd.read_csv('train.csv/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('test.csv/test.csv', encoding="ISO-8859-1")
#df_attr = pd.read_csv('attributes.csv/attributes.csv')
df_pro_desc = pd.read_csv('product_descriptions.csv')
num_train = df_train.shape[0]

def str_stemmer(s):
    if not isinstance(s, str):
        s = str(s)
    return " ".join([stemmer.stem(word) for word in s.lower().split()])

def str_common_word(str1, str2):
	return sum(int(str2.find(word)>=0) for word in str1.split())

#df_train = pd.merge(df_train, df_attributes, on='product_uid', how='left')
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True) # Concatenate the training and testing data

df_all = df_all.rename(columns={'product_uid ': 'product_uid'})
df_all = df_all.rename(columns={'product_uid ': 'product_uid'})

#print("Columns in df_all: ", df_all.columns)
#print("Columns in df_pro_desc: ", df_pro_desc.columns)

df_all = pd.merge(df_all, df_pro_desc, how='left', on="product_uid")
#df_all = pd.merge(df_all, df_attributes, how='left', on="product_uid")
if stemming == 'y':
    print("Executing with stemming")
    df_all['search_term'] = df_all['search_term'].map(lambda x:str_stemmer(x))
    df_all['product_title'] = df_all['product_title'].map(lambda x:str_stemmer(x))
    df_all['product_description'] = df_all['product_description'].map(lambda x:str_stemmer(x))
else:
     print("Executing without stemming")
df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64) # Length of the search query

df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title']+"\t"+df_all['product_description']# Concatenate the search term, product title, and product description

df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))# Number of common words between the search term and product title
df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2])) # Number of common words between the search term and product description
df_all['new_feature'] = df_all['len_of_query'] * df_all['word_in_description']
df_all['new_feature2'] = df_all['len_of_query'] * df_all['word_in_title']
df_all['new_feature3'] = df_all['word_in_description'] * df_all['word_in_title']
df_all = df_all.drop(['search_term','product_title','product_description','product_info'],axis=1)
df_train = df_all.iloc[:num_train]

# Define the features (X) and target variable (y)
X = df_train.drop(['id', 'relevance'], axis=1)  # Features
y = df_train['relevance']  # Target variable

# Split the data into training and testing sets (80% training, 20% testing)
# linear,Decision Tree Regression, k-nearest neighbors, randomforrest
#we will also measure how long it takes with every model



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameters grid for each model
hyperparameters = {
     #list(range(1,101))
    'linear': {},
    'decision_tree': {'max_depth': [6], 'min_samples_split': [53], 'min_samples_leaf': [191], 'criterion': ['squared_error']},
    'knn': {'n_neighbors': [3, 4, 5], 'weights': ['uniform', 'distance'],  'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'leaf_size': [30, 40, 50]},
    'random_forest': {'n_estimators': [20, 30, 40], 'max_depth': [10, 20, 30], 'max_features': ['sqrt', 'log2'], 'criterion': ['squared_error', 'absolute_error']},
    'svm': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
}

# Define models
models = {
    #'linear': LinearRegression(),
    'decision_tree': DecisionTreeRegressor(),
    #'knn': KNeighborsRegressor(),
    #'random_forest': RandomForestRegressor(),
    #'svm': SVR()
}
start = time.time()

best_model = None
best_rmse = float('inf')
for model_name, clf in models.items():
    params = hyperparameters[model_name]
    if params:
        grid_search = GridSearchCV(clf, params, scoring='neg_mean_squared_error', cv=5)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        best_clf = grid_search.best_estimator_
    else:
        best_clf = clf

    best_clf.fit(X_train, y_train)
    y_pred = best_clf.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"Model: {model_name}, RMSE: {rmse}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_model = best_clf
end = time.time()
print("Time taken: ", end-start)
print("Best model:", best_model)
print("Lowest RMSE:", best_rmse)



"""regression = input("Choose your regression model. Options: 1 =  linear, 2 = Decision Tree Regression, 3 = k-nearest neighbors, 4 = randomforrest:, 5= Support Vector: ")

while regression != 'stop':
    if regression == "1":
        clf = LinearRegression()#hyperparameters
    elif regression == "2":
        clf = DecisionTreeRegressor(max_depth=10,min_samples_split=3,min_samples_leaf=5,criterion='absolute_error')
    elif regression == "3":
        clf = KNeighborsRegressor()
    elif regression == "4":
        rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
        clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
    elif regression == "5":
        clf = SVR()
    else:
        print("Invalid regression model.")

    start = time.time()

    #k_folds = KFold(n_splits = 5)
    #scores = cross_val_score(clf, X, y, cv = k_folds)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    end = time.time()

    print("Time taken: ", end-start)
    print("Root Mean Squared Error (RMSE):", rmse)

    #reset clf, y_pred, rmse, start, end
    clf = None
    y_pred = None

    regression = input("Choose your regression model. Options: 1 =  linear, 2 = Decision Tree Regression, 3 = k-nearest neighbors, 4 = randomforrest: ")

"""