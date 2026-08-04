"""
Socioeconomic Drivers of Global Ecological Footprints
Author: Kaia Quigg
Description: Complete pipeline for parsing, cleaning, and imputing 
             macroeconomic sustainability data from the Global Footprint Network.
             Prepares clean tabular data for interactive Tableau BI deployment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# =====================================================================
# DATA CLEANING
# =====================================================================

df = pd.read_csv('Global Ecological Footprint 2023.csv', encoding='latin1')

df = df.rename(columns={
    'Built up land': 'Built-up land EF',
    'Built up land.1': 'Built-up land BC',
    'Total biocapacity ': 'Total Biocapacity',
    'Life Exectancy': 'Life Expectancy'
})
print(df.columns)

# Identify missing values
missing_locations = df.isnull().stack().reset_index()
missing_locations.columns = ['Country Index', 'Column', 'Is Missing']
missing_locations = missing_locations[missing_locations['Is Missing']]
missing_locations['Country'] = df.loc[missing_locations['Country Index'], 'Country'].values
missing_locations = missing_locations.drop(columns='Is Missing')
missing_locations = missing_locations[['Country', 'Column', 'Country Index']]
unique_countries_with_missing_values = missing_locations['Country'].unique()

print(f'Number of unique countries with missing values: {len(unique_countries_with_missing_values)}')
print('\nLocations of missing values:')
print(missing_locations['Column'].value_counts())

land_type_columns = [
    'Grazing land', 'Grazing Footprint', 'Fish Footprint', 'Cropland Footprint',
    'Built-up land BC', 'Forest Product Footprint', 'Carbon Footprint', 
    'Cropland', 'Built-up land EF', 'Forest land', 'Fishing ground'
]
land_type_missing = missing_locations[missing_locations['Column'].isin(land_type_columns)]
countries_of_interest = land_type_missing['Country'].unique()
print('\n'.join(countries_of_interest))

# Examining which region countries belong to
filtered_df = df[df['Country'].isin(countries_of_interest)]
region_counts = filtered_df['Region'].value_counts()
print(region_counts)

region_counts_original = df['Region'].value_counts()
print(region_counts_original)

print(df['Country'].count())
df = df[~df['Country'].isin(countries_of_interest)]
print(df['Country'].count())

# Imputing regional median to remaining missing values
def convert_gdp_to_float(df, col):
    df[col] = df[col].astype(str)  
    df[col] = df[col].str.strip()  
    df[col] = df[col].replace('[\$,]', '', regex=True)  
    df[col] = pd.to_numeric(df[col], errors='coerce')  
    return df

def identify_non_numeric(df, cols):
    non_numeric = {}
    for col in cols:
        non_numeric_values = df[~df[col].apply(lambda x: pd.to_numeric(x, errors='coerce')).notna()][col]
        if not non_numeric_values.empty:
            non_numeric[col] = non_numeric_values
    return non_numeric

def fill_with_regional_median(df, group_col, target_cols):
    for col in target_cols:
        df[col] = df.groupby(group_col)[col].transform(lambda x: x.fillna(x.median()))
    return df

df = convert_gdp_to_float(df, 'Per Capita GDP')
numerical_columns = ['Per Capita GDP', 'SDGi']  
non_numeric_values = identify_non_numeric(df, numerical_columns)
for col, values in non_numeric_values.items():
    print(f"Non-numeric values in column {col}:")
    print(values)

for col in numerical_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = fill_with_regional_median(df, 'Region', numerical_columns)

missing_values = df.isnull().sum()
print(missing_values)

df['Life Expectancy'] = df['Life Expectancy'].astype(str)
df['HDI'] = df['HDI'].astype(str)
df['Life Expectancy'] = df['Life Expectancy'].str.strip()  
df['Life Expectancy'] = df['Life Expectancy'].replace('', np.nan)
df['HDI'] = df['HDI'].str.strip() 
df['HDI'] = df['HDI'].replace('', np.nan) 
df['Life Expectancy'] = pd.to_numeric(df['Life Expectancy'], errors='coerce')
df['HDI'] = pd.to_numeric(df['HDI'], errors='coerce')

remaining_numerical_cols = ['Life Expectancy', 'HDI']
df = fill_with_regional_median(df, 'Region', remaining_numerical_cols)
missing_values_after_fill = df.isnull().sum()
print("Missing values after filling with regional median:")
print(missing_values_after_fill)

income_mapping = {'HI': 3, 'LI': 0, 'LM': 1, 'UM': 2}
df['Income Group'] = df['Income Group'].map(income_mapping)
print(df[['Income Group']])

# =====================================================================
# Research Question 1
# =====================================================================

# Summary statistics
columns_of_interest = ['Total Ecological Footprint (Consumption)', 'Total Biocapacity', 'Per Capita GDP', 'HDI', 'Income Group']
summary_stats = df[columns_of_interest].agg(['mean', 'median', 'std', lambda x: x.max() - x.min()]).transpose()
summary_stats.columns = ['Mean', 'Median', 'Standard Deviation', 'Range']
print(summary_stats)

# Correlation
corr = df[columns_of_interest].corr(method = 'spearman')
print(corr)

# Regression
X = df[['Per Capita GDP', 'HDI']]
y = df['Total Ecological Footprint (Consumption)']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

# Graphics
# HDI vs Total Ecological Footprint
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='HDI', y='Total Ecological Footprint (Consumption)', hue='Income Group', palette='viridis')
plt.title('HDI vs Total Ecological Footprint (Consumption)')
plt.xlabel('HDI')
plt.ylabel('Total Ecological Footprint (Consumption)')
plt.legend(title='Income Group', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# HDI vs Total Biocapacity
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='HDI', y='Total Biocapacity', hue='Income Group', palette='viridis')
plt.title('HDI vs Total Biocapacity')
plt.xlabel('HDI')
plt.ylabel('Total Biocapacity')
plt.legend(title='Income Group', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# =====================================================================
# Research Question 2
# =====================================================================

# Regional Differences in resource use
# Avg EF by region by land type
avg_eco_footprints = df.groupby('Region').mean()[[
    'Cropland Footprint', 'Grazing Footprint', 
    'Forest Product Footprint', 
    'Fish Footprint', 'Built-up land EF']]

# Avg BC by region by land type
avg_biocapacity = df.groupby('Region').mean()[[
    'Cropland', 'Grazing land', 
    'Forest land', 'Fishing ground', 
    'Built-up land BC']]

print("Average Ecological Footprints by Region:")
print(avg_eco_footprints)
print("\nAverage Biocapacity by Region:")
print(avg_biocapacity)

# Data exported to Tableau for further manipulation/visualization
avg_eco_footprints.to_csv('average_ecological_footprints.csv', index=True)
avg_biocapacity.to_csv('average_biocodingcapacity.csv', index=True)

# Normalizing by population
df['Population (millions)'] = pd.to_numeric(df['Population (millions)'], errors='coerce')
df['Total Cropland Footprint'] = df['Cropland Footprint'] * df['Population (millions)']
df['Total Grazing Footprint'] = df['Grazing Footprint'] * df['Population (millions)']
df['Total Forest Product Footprint'] = df['Forest Product Footprint'] * df['Population (millions)']
df['Total Fish Footprint'] = df['Fish Footprint'] * df['Population (millions)']
df['Total Built-up Land Footprint'] = df['Built-up land EF'] * df['Population (millions)']

df['Total Cropland Biocapacity'] = df['Cropland'] * df['Population (millions)']
df['Total Grazing Biocapacity'] = df['Grazing land'] * df['Population (millions)']
df['Total Forest Biocapacity'] = df['Forest land'] * df['Population (millions)']
df['Total Fishing Biocapacity'] = df['Fishing ground'] * df['Population (millions)']
df['Total Built-up Land Biocapacity'] = df['Built-up land BC'] * df['Population (millions)']

# Calculate differences for each resource type
df['Difference Cropland'] = df['Total Cropland Biocapacity'] - df['Total Cropland Footprint']
df['Difference Grazing'] = df['Total Grazing Biocapacity'] - df['Total Grazing Footprint']
df['Difference Forest'] = df['Total Forest Biocapacity'] - df['Total Forest Product Footprint']
df['Difference Fishing'] = df['Total Fishing Biocapacity'] - df['Total Fish Footprint']
df['Difference Built-up Land'] = df['Total Built-up Land Biocapacity'] - df['Total Built-up Land Footprint']

# Group by region and aggregate differences
regional_differences = df.groupby('Region').agg({
    'Difference Cropland': 'sum',
    'Difference Grazing': 'sum',
    'Difference Forest': 'sum',
    'Difference Fishing': 'sum',
    'Difference Built-up Land': 'sum',
}).reset_index()

# Print aggregated regional differences to compare
print(regional_differences)

# Exported to Tableau
regional_differences.to_csv('regional_differences.csv', index=False)

# =====================================================================
# Research Question 3
# =====================================================================

# Scatter plots to visualize predictors
sns.pairplot(df, x_vars=['Per Capita GDP', 'HDI', 'Income Group'], y_vars='Total Ecological Footprint (Consumption)', height=5, aspect=0.7)
plt.show()

# Build model
X = df[['Per Capita GDP', 'HDI', 'Income Group']]
y_ef = df['Total Ecological Footprint (Consumption)']
X = pd.get_dummies(X, drop_first=True)
X = sm.add_constant(X)

# Split data
X_train_ef, X_test_ef, y_train_ef, y_test_ef = train_test_split(X, y_ef, test_size=0.2, random_state=12)

# Run model
model_ef = sm.OLS(y_train_ef, X_train_ef).fit()
print(model_ef.summary())

# Predict and evaluate
y_pred_ef = model_ef.predict(X_test_ef)
print(f'EF Model R^2: {r2_score(y_test_ef, y_pred_ef)}')
print(f'EF Model RMSE: {np.sqrt(mean_squared_error(y_test_ef, y_pred_ef))}')

# Visualize # Actual vs Predicted values for EF
plt.figure(figsize=(10, 5))
plt.plot(y_test_ef.values, label='Actual EF')
plt.plot(y_pred_ef.values, label='Predicted EF')
plt.legend()
plt.title('Actual vs Predicted Ecological Footprint')
plt.show()

# Residual plots for EF
residuals_ef = y_test_ef - y_pred_ef
sns.residplot(x=y_pred_ef, y=residuals_ef, lowess=True, color='g')
plt.title('Residuals of Ecological Footprint Model')
plt.show()
