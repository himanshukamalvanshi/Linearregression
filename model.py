import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('baseball_players.csv')
df.head()

average_value=df['Weight(pounds)'].mean()
df['Weight(pounds)'].fillna(average_value, inplace=True)

x = df[['Height(inches)']].values
y = df['Weight(pounds)'].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

pickle.dump(regressor, open('model.pkl', 'wb'))