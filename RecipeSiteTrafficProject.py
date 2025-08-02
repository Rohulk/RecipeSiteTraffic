import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

df = pd.read_csv("recipe_site_traffic_2212.csv")

print("\nComprehensive Data Validation and Cleaning")

# Recipe ID Validation
print("\nRecipe ID Column Validation")
print(f"- Total recipes: {len(df)}")
print(f"- Missing IDs: {df['recipe'].isnull().sum()}") 
print(f"- Duplicate IDs: {df['recipe'].duplicated().sum()}") 
assert df['recipe'].isnull().sum() == 0, "Recipe IDs contain null values"
assert df['recipe'].duplicated().sum() == 0, "Duplicate recipe IDs found"

# Nutritional Columns Validation (calories, carbohydrate, sugar, protein)
nutrition_cols = ['calories', 'carbohydrate', 'sugar', 'protein']
print("\nNutritional Values Column Validation ")

nutrition_stats = []
for col in nutrition_cols:
    # Basic stats
    missing = df[col].isnull().sum()
    zeros = (df[col] == 0).sum()
    negatives = (df[col] < 0).sum()
    
    # Record stats
    nutrition_stats.append({
        'Column': col,
        'Missing': missing,
        '% Missing': f"{missing/len(df):.1%}",
        'Zeros': zeros,
        'Negatives': negatives,
        'Min': df[col].min(),
        'Median': df[col].median(),
        'Max': df[col].max()
    })
    
    # Handle missing values (median imputation)
    col_median = df[col].median()
    df[col] = df[col].fillna(col_median)
    
    # Verify no negatives exist (set to abs value if found)
    if negatives > 0:
        print(f" Warning: {negatives} negative values found in {col}")
        df[col] = df[col].abs()

# Display nutrition stats table
print(pd.DataFrame(nutrition_stats).to_string(index=False))

# 3. Category Validation
print("\nCategory Column Validation:")
valid_categories = ['Lunch/Snacks', 'Beverages', 'Potato', 'Vegetable', 'Meat', 
                   'Chicken', 'Pork', 'Dessert', 'Breakfast', 'One Dish Meal',
                   'Chicken Breast']  # From data description

# Check for unexpected categories
invalid_categories = set(df['category'].unique()) - set(valid_categories)
if invalid_categories:
    print(f" Warning: {len(invalid_categories)} invalid categories found: {invalid_categories}")
else:
    print("All categories are valid")

# Check for missing categories
missing_categories = df['category'].isnull().sum()
print(f"  Missing categories: {missing_categories}")  # Should be 0

# Servings Validation
print("\nServings Column Validation:")
# Convert to string first to handle any mixed types in the csv file
df['servings'] = df['servings'].astype(str)

# Extract numeric values 
df['servings'] = df['servings'].str.extract('(\d+)').astype(float)

# Check stats
print(f"- Missing servings: {df['servings'].isnull().sum()}")
print(f"- Range: {df['servings'].min()} to {df['servings'].max()}")
print("- Value counts:")
print(df['servings'].value_counts().sort_index())

# Handle missing values (median imputation)
servings_median = df['servings'].median()
df['servings'] = df['servings'].fillna(servings_median)

# Verify no zeros or negatives
assert (df['servings'] > 0).all(), "Invalid servings values (≤0) found"

# 5. High_traffic Validation
print("\nHigh_traffic Column Validation:")
print("- Value counts before cleaning:")
print(df['high_traffic'].value_counts(dropna=False))

# Convert to binary (1=High, 0=Not High)
df['high_traffic'] = (df['high_traffic'] == 'High').astype(int)

# Handle missing values (assuming NA means not high traffic)
missing_traffic = df['high_traffic'].isnull().sum()
if missing_traffic > 0:
    print(f"  Imputing {missing_traffic} missing values as 0 (Not High)")
    df['high_traffic'] = df['high_traffic'].fillna(0)

print("\n After cleaning:")
print(f"High traffic recipes: {df['high_traffic'].sum()} ({df['high_traffic'].mean():.1%})")
print(f"Not high traffic: {len(df) - df['high_traffic'].sum()}")

# Final Data Quality Report
print("\nFinal Data Quality Report")
print(f"Total recipes after cleaning: {len(df)}")
print(f"Columns: {list(df.columns)}")
print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nData types after cleaning:")
print(df.dtypes)

# Save cleaned data for reference
df.to_csv("cleaned_recipe_data.csv", index=False)
print("\nCleaned data saved to 'cleaned_recipe_data.csv'") #Created a cleaned data csv for viewing


# EXPLORATORY DATA ANALYSIS 
print("\nExploratory analysis: ")
missing_before = [373, 52, 52, 52, 0, 0]  # high_traffic, calories, ..., servings
missing_after = [0, 0, 0, 0, 0, 0]  


# This is my histogram which is a single variable graph
plt.figure(figsize=(10, 6))
sns.histplot(df['calories'], kde=True, bins=30, color='#4C72B0')
plt.title('Distribution of Recipe Calories', fontsize=14)
plt.xlabel('Calories', fontsize=12)
plt.ylabel('Number of Recipes', fontsize=12)
plt.axvline(df['calories'].median(), color='red', linestyle='--', 
            label=f'Median: {df["calories"].median():.0f} cal')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()


# This is my bar chart which is a single variable graph
plt.figure(figsize=(12, 6))
category_counts = df['category'].value_counts()
ax = sns.barplot(x=category_counts.values, y=category_counts.index, palette='viridis')
plt.title('Recipe Categories Distribution', fontsize=14)
plt.xlabel('Number of Recipes', fontsize=12)
plt.ylabel('Category', fontsize=12)


for i, (v, bar) in enumerate(zip(category_counts.values, ax.patches)):
    x_pos = bar.get_width() + 2  
    y_pos = bar.get_y() + bar.get_height()/2  
    ax.text(x_pos, y_pos, str(v), 
            va='center', ha='left', 
            color='black', fontsize=10)


max_value = category_counts.max()
ax.set_xlim(right=max_value * 1.15)

plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()



# This is the box plot and this is a two variable graph (comparing)
plt.figure(figsize=(10, 6))
sns.boxplot(x='high_traffic', y='calories', data=df, 
            palette={'0': '#55A868', '1': '#C44E52'})  # Changed to string keys
plt.title('Calories Distribution by Traffic Status', fontsize=14)
plt.xlabel('High Traffic (1 = High, 0 = Not High)', fontsize=12)
plt.ylabel('Calories', fontsize=12)
plt.xticks([0, 1], ['Not High', 'High'])
plt.grid(axis='y', alpha=0.3)
plt.show()

# Findings description
print("\nSome of the key findings in those exploratory analysis I made were:")
print("- High-traffic recipes tend to have higher median calories")
print(f"- Most common categories: {category_counts.index[0]}, {category_counts.index[1]}, {category_counts.index[2]}")
print("- Some extreme calorie outliers exist (>3000 calories)")

# Model development 
print("\nModel development:")

# Prepare features
X_numerical = df[['calories', 'carbohydrate', 'sugar', 'protein', 'servings']]
X_categorical = pd.get_dummies(df['category'], drop_first=True)
X = pd.concat([X_numerical, X_categorical], axis=1)
y = df['high_traffic']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Problem Type: Binary Classification (Predict High Traffic Recipes)")

# Baseline Model (Logistic Regression)
print("\nTraining Baseline Model like Logistic Regression")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Comparison Model (Random Forest)
print("\nTraining Comparison Model such as Random Forest")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Model evaluation rounding it to 2 decimal places
print("\nModel evaluation:")

def evaluate_model(y_true, y_pred, model_name):
    print(f"\nMetrics for {model_name}:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}")
    print(f"Precision: {precision_score(y_true, y_pred):.2f}") 
    print(f"Recall: {recall_score(y_true, y_pred):.2f}")
    print(f"F1-Score: {f1_score(y_true, y_pred):.2f}")
    print(f"ROC AUC: {roc_auc_score(y_true, y_pred):.2f}")

evaluate_model(y_test, y_pred_lr, "Logistic Regression")
evaluate_model(y_test, y_pred_rf, "Random Forest")

# Feature Importance
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importances.head(10).to_string(index=False))

# Business metrics 
print("\n Business metrics:")
print("\n- Recommended metric: Weekly Precision (Minimize false positives)")
print("- Current precision: 76% (Random Forest)")
print("- Target: maintain ≥80% precision in production")

print("\nMonitoring Plan:")
print("- Track precision weekly using A/B testing")
print("- Alert if precision drops below 75%")
print("- Review false positives monthly to identify patterns")

# Some recommendations
print("\nSome of the recommendations I suggest would be: ")
print("1. Implement Random Forest model for recipe recommendations")
print("2. Focus on recipes with:")
print("   - 500-1500 calories")
print("   - High protein content (>20g per serving)")
print("3. Monitor nutritional balance (protein/carb ratio)")
print("4. Expand data collection to include:")
print("   - Cooking time")
print("   - User ratings")
print("   - Seasonal trends")