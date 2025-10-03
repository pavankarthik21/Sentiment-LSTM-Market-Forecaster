import pandas as pd

df = pd.read_csv("output" + "gs"+".csv")
df2 = df.iloc[::-1]
df2.reset_index(inplace = True, )
print(df2)






