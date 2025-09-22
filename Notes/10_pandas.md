# 10 Pandas Cheatsheet

This cheatsheet covers common Pandas operations for data manipulation and analysis in Python.

## Importing Pandas
```python
import pandas as pd
```

## Creating DataFrames
```python
# From a dictionary
data = {'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}
df = pd.DataFrame(data)

# From a list of dictionaries
data = [{'col1': 1, 'col2': 'a'}, {'col1': 2, 'col2': 'b'}]
df = pd.DataFrame(data)

# From a CSV file
df = pd.read_csv('file.csv')
```

## Inspecting Data
```python
# View first 5 rows
df.head()

# View last 5 rows
df.tail()

# Get basic info
df.info()

# Get summary statistics
df.describe()

# Shape of DataFrame (rows, columns)
df.shape
```

## Selecting Data
```python
# Select a column
df['col1']

# Select multiple columns
df[['col1', 'col2']]

# Select rows by index
df.iloc[0:2]  # First two rows

# Select rows by label
df.loc['index_label']

# Conditional selection
df[df['col1'] > 1]
```

## Data Manipulation
```python
# Add a new column
df['new_col'] = df['col1'] * 2

# Drop a column
df.drop('col1', axis=1, inplace=True)

# Drop rows with missing values
df.dropna()

# Fill missing values
df.fillna(value=0)

# Sort by column
df.sort_values('col1', ascending=True)

# Reset index
df.reset_index(drop=True)
```

## Grouping and Aggregating
```python
# Group by a column
grouped = df.groupby('col1')

# Aggregate functions
grouped.agg({'col2': ['mean', 'sum', 'count']})

# Pivot table
df.pivot_table(values='col1', index='col2', columns='col3', aggfunc='mean')
```

## Merging and Joining
```python
# Merge DataFrames
df_merged = pd.merge(df1, df2, on='key_column', how='inner')

# Concatenate DataFrames vertically
df_concat = pd.concat([df1, df2])

# Join DataFrames on index
df_joined = df1.join(df2)
```

## Filtering and Querying
```python
# Filter rows
df[df['col1'] > 10]

# Query with conditions
df.query('col1 > 10 and col2 == "value"')

# Unique values in a column
df['col1'].unique()
```

## Handling Missing Data
```python
# Check for missing values
df.isna().sum()

# Replace missing values
df.fillna({'col1': 0, 'col2': 'missing'})

# Interpolate missing values
df.interpolate()
```

## Working with Dates
```python
# Convert to datetime
df['date_col'] = pd.to_datetime(df['date_col'])

# Extract year
df['year'] = df['date_col'].dt.year

# Filter by date range
df[df['date_col'].between('2023-01-01', '2023-12-31')]
```

## Exporting Data
```python
# To CSV
df.to_csv('output.csv', index=False)

# To Excel
df.to_excel('output.xlsx', index=False)

# To JSON
df.to_json('output.json')
```

## Useful Tips
- Use `inplace=True` to modify the DataFrame directly.
- Chain operations for concise code, but ensure readability.
- Use `pd.set_option('display.max_columns', None)` to show all columns.