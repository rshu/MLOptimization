import pandas as pd
from io import StringIO
from sklearn.preprocessing import Imputer


csv_data = '''A,B,C,D
    1.0,2.0,3.0,4.0
    5.0,6.0,,8.0
    0.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))
print(df)
print()
print(df.isnull().sum())
print()
print(df.dropna())
print()
print(df.dropna(axis=1))
print()
# only drop rows where all columns are NaN
# df.dropna(how='all')
# drop rows that have not at least 4 non-NaN values
# df.dropna(thresh=4)
# only drop rows where NaN appear in specific columns (here: 'C')
# df.dropna(subset=['C'])

imr = Imputer(missing_values='NaN', strategy='mean', axis=0) # mean of column
# imr = Imputer(missing_values='NaN', strategy='mean', axis=1) # mean of row
# strategy='median', strategy='most_frequent'
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
print(imputed_data)
