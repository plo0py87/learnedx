import pandas as pd
import matplotlib.pyplot as plt
s=pd.read_excel("uncertainty.xlsx", 'sheet1', skiprows = 3, nrows=5, usecols = 'A:F')
print(s.head())
f=s["1st"].tolist()
print(type(s['1st']))
plt.scatter(f,[1,2,3,4,5])
plt.show()
print(s.loc[1])
print(f)
# plt.scatter(s[])
# plt.show()
