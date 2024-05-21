# Data handling
import pandas as pd  # noqa
import numpy as np  # noqa
import matplotlib.pyplot as plt

train = pd.read_csv(
    'train.csv/train.csv'
)

print(train.shape[0]) #to show the number of rows in the dataset

print(train.iloc[:, 1].nunique()) #to show the unique values in the second column

print("Most Common two values are:", train.iloc[:, 1].value_counts().head(2)) #to show the frequency of most two common unique value in the second column

mean = train.iloc[:, 4]#to show the mean of the fourth column
median = train.iloc[:, 4]#to show the median of the fourth column
std_dev = train.iloc[:, 4]#to show the standard deviation of the fourth column

print("Mean of the fourth column is:", mean.mean())
print("Median of the fourth column is:", median.median())
print("Standard Deviation of the fourth column is:", std_dev.std())

train.boxplot(column='relevance') #to show the boxplot of the fourth column
plt.title('Boxplot of relevance')
plt.ylabel('Values')
plt.show()

attributes = pd.read_csv(
    'attributes.csv/attributes.csv'
)

print("Most common 5 brand names are:", attributes.iloc[:, 1].value_counts().head(5)) #to show the frequency of most common 5 brand names
