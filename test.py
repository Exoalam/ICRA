import pandas as pd

# Step 1: Read data from a CSV file
df = pd.read_csv('hybrid_reg.csv')

# Step 2: Remove rows where column 'B' is 0 (just as an example transformation)
df = df[df['x1'] != 0]

# Step 3: Write the modified data to a new CSV file
df.to_csv('output.csv', index=False)  # 'index=False' means don't save row numbers
