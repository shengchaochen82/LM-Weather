import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('dataset/Weather-Tiny/114.csv')
# Assume that df is your DataFrame
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Plotting each column
fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(15, 10))
for i, column in enumerate(df.columns):
    df[column].plot(ax=axes[i//3, i%3])
    axes[i//3, i%3].set_title(column)

fig.tight_layout()
plt.show()