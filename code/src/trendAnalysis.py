import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import os
import pickle
from statsmodels.stats.outliers_influence import variance_inflation_factor
from dataCleanup import cleanup_data
from sklearn.model_selection import train_test_split

raw_data = pl.read_csv('../../data/kaggle_car_prices.csv')

df = cleanup_data(raw_data)

df = df.with_columns(
    (pl.col("odometer") / 10000).alias("odometer_10k")
)

# Setup a 1x3 grid for the 3 plots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Automobile Sales Trend Analysis', fontsize=16)

# Plot 1: Most common makes being sold (Top 15)
top_makes = df['make'].value_counts().sort('count', descending=True).head(15).to_pandas()
sns.barplot(data=top_makes, y='make', x='count', hue='make', legend=False, ax=axes[0], palette='magma')
axes[0].set_title('Top 15 Most Common Makes Sold')
axes[0].set_xlabel('Number of Vehicles Sold')
axes[0].set_ylabel('Make')

# Plot 2: Most common years of vehicle sold
common_years = df['year'].value_counts().sort('count', descending=True).head(20).sort('year').to_pandas()
sns.barplot(data=common_years, x='year', y='count', hue='year', legend=False, ax=axes[1], palette='crest')
axes[1].set_title('Most Common Vehicle Years Sold (Top 20)')
axes[1].set_xlabel('Vehicle Year')
axes[1].set_ylabel('Number of Vehicles Sold')
axes[1].tick_params(axis='x', rotation=45)

# Plot 3: Comparison of vehicle age to selling price
age_price = df.group_by('vehicle_age').agg(
    pl.col('sellingprice').mean().alias('mean_price')
).sort('vehicle_age')
# Filter to a reasonable age range to avoid outliers skewing the plot
age_price = age_price.filter(pl.col('vehicle_age') <= 30).to_pandas()

sns.lineplot(data=age_price, x='vehicle_age', y='mean_price', ax=axes[2], marker='o', color='b')
axes[2].set_title('Average Selling Price by Vehicle Age')
axes[2].set_xlabel('Vehicle Age (Years)')
axes[2].set_ylabel('Average Selling Price ($)')

plt.tight_layout()
plt.savefig('trend_analysis.png')
print("Plots successfully generated and saved to trend_analysis.png")

y = df["sellingprice"]

features = [
    "vehicle_age",
    "odometer_10k",
    "condition",
    # "mmr", # We need to remove mmr as it's already an estimate for the price of the vehicle
    # "make", # make is captured in model since a model like z3 is only for BMW
    "model",
    "state_grouped",  # Added state to predict price variations by location
    # "body", # Body tends to be captured by the model as well
    # "transmission", # insignificant
    # "color", # color is not statistically significant and mostly just added noise
]

# Convert to pandas
pdf = df.to_pandas()

# Take a stratified sample of 5000 rows
sample_pdf, _ = train_test_split(
    pdf,
    train_size=5000,
    random_state=42,
    stratify=pdf["state"]
)

# Split sampled data into train/test
train_pdf, test_pdf = train_test_split(
    sample_pdf,
    test_size=0.2,
    random_state=42
)

# Tricky, some models are very rare and add a lot of extra variables to look at. With so few items it may start to memorize the data as well,
# so we will combine rare models together, while preserving the make still so like BMW_Other.
threshold = 20

model_counts = (
    train_pdf.groupby(['make', 'model'])
             .size()
             .reset_index(name='count')
)

common_models = set(
    zip(
        model_counts.loc[model_counts['count'] >= threshold, 'make'],
        model_counts.loc[model_counts['count'] >= threshold, 'model']
    )
)

def collapse_model(row):
  if (row['make'], row['model']) in common_models:
    return row['model']
  return f"{row['make']}_Other"

train_pdf['model'] = train_pdf.apply(collapse_model, axis=1)
test_pdf['model'] = test_pdf.apply(collapse_model, axis=1)

# Group states that don't have very many observations as well.
state_threshold = 30

state_counts = train_pdf["state"].value_counts()
kept_states = state_counts[state_counts >= state_threshold].index

def group_state(x):
    return x if x in kept_states else "Other"

train_pdf["state_grouped"] = train_pdf["state"].apply(group_state)
test_pdf["state_grouped"] = test_pdf["state"].apply(group_state)

X_train = pd.get_dummies(
    train_pdf[features],
    drop_first=True
)

X_test = pd.get_dummies(
    test_pdf[features],
    drop_first=True
)

X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)

X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

y_train = np.log(train_pdf["sellingprice"])

X_train = X_train.astype(float)
X_test = X_test.astype(float)

y_train = y_train.astype(float)


print(X_train.dtypes)

print(X_train.dtypes.value_counts())

fit_log = sm.OLS(y_train, X_train).fit()

print(fit_log.summary())

y_test = np.log(test_pdf["sellingprice"])

y_test = y_test.astype(float)

predictions = fit_log.predict(X_test)

residuals = y_test - predictions

plt.figure(figsize=(10, 6))
plt.scatter(predictions, residuals)
plt.axhline(0)
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted")
plt.savefig('ResidualsVsFitted.png')

sm.qqplot(residuals, line='45')

plt.title("Q-Q Plot")
plt.savefig('qqPlot.png')

vif_data = pd.DataFrame({
    "Variable": X_train.columns,
    "VIF": [variance_inflation_factor(X_train.values, i)
            for i in range(X_train.shape[1])]
})

print(
    vif_data.sort_values("VIF", ascending=False)
)

train_pdf["log_price"] = np.log(train_pdf["sellingprice"])

plt.figure(figsize=(10, 6))

sns.regplot(
    data=train_pdf,
    x="vehicle_age",
    y="log_price",
    scatter_kws={"alpha": 0.15},
    line_kws={"linewidth": 3}
)

plt.title("Vehicle Age vs Selling Price")
plt.xlabel("Vehicle Age (Years)")
plt.ylabel("Selling Price (log $)")
plt.tight_layout()
plt.savefig('ageVsSellingPrice.png')

plt.figure(figsize=(10, 6))

sns.regplot(
    data=train_pdf,
    x="odometer_10k",
    y="log_price",
    scatter_kws={"alpha": 0.15},
    line_kws={"linewidth": 3}
)

plt.title("odometer (10k) vs Selling Price")
plt.xlabel("odometer (10k) Reading")
plt.ylabel("Selling Price (log $)")
plt.tight_layout()
plt.savefig('OdometerVsSellingPrice.png')

# Construct confidence interval data frame
df_01 = pl.DataFrame({
    'term': fit_log.conf_int().index.tolist(),
    'coef': fit_log.params.tolist(),
    'conf_low': fit_log.conf_int().loc[:, 0].tolist(),
    'conf_high': fit_log.conf_int().loc[:, 1].tolist()
})

# Selecting just slopes
df_02 = df_01.filter(pl.col('term') != 'const')
df_03 = df_02.filter(
    pl.col('term').str.contains('model'),
    ~pl.col('term').str.contains('Other')
)

# Plotting the confidence interval
dfm = (
    df_03
    .sort('coef')
)

n = len(dfm)

plt.figure(figsize=(12, max(12, n * 0.25)))
plt.errorbar(
    dfm['coef'],
    dfm['term'],
    xerr=[dfm['coef'] - dfm['conf_low'], dfm['conf_high'] - dfm['coef']],
    fmt='o',
    capsize=2,
    markersize=3,
    elinewidth=1
)
plt.axvline(0, color='red', linestyle='--')
plt.gca().invert_yaxis()
plt.yticks(fontsize=8)
plt.subplots_adjust(left=0.35)
plt.savefig('KnownModelsEffects.png', bbox_inches='tight', dpi=300)

df_04 = df_02.filter(
    pl.col('term').str.contains('model'),
    pl.col('term').str.contains('Other')
)

# Plotting the confidence interval
df = df_04
plt.figure(figsize=(18, 9))
plt.errorbar(df['coef'], df['term'],
    xerr=[df['coef'] - df['conf_low'], df['conf_high'] - df['coef']], 
    fmt='o', 
    capsize=5, 
    label='Estimates')
plt.axvline(0, color='red', linestyle='--', label='y=0')
plt.savefig('OtherModelEffects.png')

df_03 = df_02.filter(~pl.col('term').str.contains('model'),
                     ~pl.col('term').str.contains('state'))

# Plotting the confidence interval
df = df_03
plt.figure(figsize=(10, 3))
plt.errorbar(df['coef'], df['term'],
    xerr=[df['coef'] - df['conf_low'], df['conf_high'] - df['coef']], 
    fmt='o', 
    capsize=5, 
    label='Estimates')
plt.axvline(0, color='red', linestyle='--', label='y=0')
plt.savefig('ParamsEffectLimited.png')

# --- Save Model and Metadata ---

output_dir = '../../output'
os.makedirs(output_dir, exist_ok=True)

# Save the trained statsmodels Results object
model_path = os.path.join(output_dir, 'car_price_model.pkl')
fit_log.save(model_path)
print(f"Trained model saved to: {model_path}")

# Save the metadata required for preprocessing and alignment
metadata_path = os.path.join(output_dir, 'model_metadata.pkl')
metadata = {
    'common_models': common_models,
    'train_columns': X_train.columns.tolist(),
    'features': features,
    'residual_variance': fit_log.scale  # Save residual variance (scale) for expectation correction
}

with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)
print(f"Model metadata saved to: {metadata_path}")
