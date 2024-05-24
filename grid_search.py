import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from nltk.stem.snowball import SnowballStemmer
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV

# Initialize Snowball Stemmer
stemmer = SnowballStemmer('english')
stemming = input("Do you want to use stemming? (y/n): ")

# Load datasets
df_train = pd.read_csv('train.csv/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('test.csv/test.csv', encoding="ISO-8859-1")
df_pro_desc = pd.read_csv('product_descriptions.csv/product_descriptions.csv')
num_train = df_train.shape[0]

# Stemming function
def str_stemmer(s):
    if not isinstance(s, str):
        s = str(s)
    return " ".join([stemmer.stem(word) for word in s.lower().split()])

# Common word count function
def str_common_word(str1, str2):
    return sum(int(str2.find(word) >= 0) for word in str1.split())

# Merge datasets
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_pro_desc, how='left', on="product_uid")

# Apply stemming if required
if stemming.lower() == 'y':
    print("Executing with stemming")
    df_all['search_term'] = df_all['search_term'].map(lambda x: str_stemmer(x))
    df_all['product_title'] = df_all['product_title'].map(lambda x: str_stemmer(x))
    df_all['product_description'] = df_all['product_description'].map(lambda x: str_stemmer(x))
else:
    print("Executing without stemming")

# Feature engineering
df_all['len_of_query'] = df_all['search_term'].map(lambda x: len(x.split())).astype(np.int64)
df_all['product_info'] = df_all['search_term'] + "\t" + df_all['product_title'] + "\t" + df_all['product_description']
df_all['word_in_title'] = df_all['product_info'].map(lambda x: str_common_word(x.split('\t')[0], x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(lambda x: str_common_word(x.split('\t')[0], x.split('\t')[2]))
df_all['new_feature'] = df_all['len_of_query'] * df_all['word_in_description']
df_all['new_feature2'] = df_all['len_of_query'] * df_all['word_in_title']
df_all['new_feature3'] = df_all['word_in_description'] * df_all['word_in_title']
df_all = df_all.drop(['search_term', 'product_title', 'product_description', 'product_info'], axis=1)
df_train = df_all.iloc[:num_train]

# Splitting data
X = df_train.drop(['id', 'relevance'], axis=1)
y = df_train['relevance']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model creation function
def create_model(optimizer='adam', init='he_uniform'):
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], kernel_initializer=init, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, kernel_initializer=init, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer=init))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

# Create and configure the KerasRegressor
model = KerasRegressor(model=create_model, verbose=0)

# Define the grid search parameters
param_grid = {
    'batch_size': [4, 8],
    'epochs': [10, 20],
    'optimizer': ['adam', 'sgd'],
}

# Perform Grid Search
start = time.time()
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, scoring='neg_mean_squared_error')
grid_result = grid.fit(X_train, y_train)
end = time.time()

# Display results
print("Grid search results:")
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    rmse = np.sqrt(-mean)
    print(f"RMSE: {rmse:.4f}, Stdev: {stdev:.4f} with: {param}")

print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
best_model = grid_result.best_estimator_

# Evaluate the best model
y_pred = best_model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
best_params = grid_result.best_params_
print(f"Neural Network RMSE: {rmse} with best parameters: {best_params}")
print("Time taken: ", end - start)




# Prepare data for heatmap
results = pd.DataFrame(grid_result.cv_results_)

# Pivot the results DataFrame for heatmap
heatmap_data = results.pivot_table(index='param_batch_size', columns='param_epochs', values='mean_test_score')

# Convert negative MSE to RMSE
heatmap_data = np.sqrt(-heatmap_data)

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="viridis")
plt.title("Grid Search RMSE for Different Hyperparameter Combinations")
plt.xlabel("Number of Epochs")
plt.ylabel("Batch Size")
plt.show()