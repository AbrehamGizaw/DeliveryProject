import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
completed = pd.read_csv('data/nb.csv')
order_data = pd.read_csv('data/driver_locations_during_request.csv')
# Basic statistics
print(df.describe())

# Missing values
print(df.isnull().sum())

# Plot distributions
sns.histplot(df['value'])
plt.show()
