import pandas as pd

# Load datasets
train = pd.read_csv("/data/train.csv")
test = pd.read_csv("/data/item_outlet_sales_output.csv")

# Display basic info
print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Preview data
print(train.head())
