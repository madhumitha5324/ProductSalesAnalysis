# ============================================================================
# COMPREHENSIVE PANDAS DATA ANALYSIS MINI PROJECT
# Sales Data Analysis using Large CSV Dataset
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set display options for better output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

print("="*80)
print("COMPREHENSIVE PANDAS DATA ANALYSIS PROJECT")
print("Sales Dataset Analysis with Complete Pandas Operations")
print("="*80)

# ============================================================================
# STEP 1: CREATING AND LOADING DATA FROM CSV
# ============================================================================

# Generate the dataset
np.random.seed(42)
n_records = 5000

# Create date range
dates = pd.date_range(start='2020-01-01', end='2023-12-31', periods=n_records)

# Product and region data
products = np.random.choice([
    'Laptop', 'Desktop', 'Mouse', 'Keyboard', 'Monitor', 'Webcam', 
    'Headphones', 'Tablet', 'Phone', 'Printer', 'Scanner', 'Speaker'
], n_records)

regions = np.random.choice(['North', 'South', 'East', 'West', 'Central'], n_records)
salespersons = np.random.choice([f'Sales_Person_{i:02d}' for i in range(1, 26)], n_records)
customers = np.random.choice([f'Customer_{i:04d}' for i in range(1, 1001)], n_records)

# Price mapping for realistic data
base_prices = {
    'Laptop': 800, 'Desktop': 600, 'Monitor': 300, 'Tablet': 400, 'Phone': 500,
    'Printer': 200, 'Scanner': 150, 'Speaker': 100, 'Mouse': 25, 
    'Keyboard': 50, 'Webcam': 75, 'Headphones': 80
}

# Create comprehensive dataset
data_records = []
for i in range(n_records):
    product = products[i]
    base_price = base_prices[product]
    unit_price = base_price * np.random.uniform(0.8, 1.5)
    quantity = np.random.randint(1, 20) if product in ['Mouse', 'Keyboard'] else np.random.randint(1, 6)
    discount = np.random.uniform(0, 0.25)
    
    subtotal = quantity * unit_price
    discount_amount = subtotal * discount
    total_amount = subtotal - discount_amount
    
    # Seasonal adjustment
    month = dates[i].month
    if month in [11, 12]:  # Holiday season boost
        total_amount *= 1.2
    
    record = {
        'Date': dates[i],
        'Product': product,
        'Category': 'Electronics' if product in ['Laptop', 'Desktop', 'Monitor', 'Tablet', 'Phone'] else 'Accessories',
        'Region': regions[i],
        'Salesperson': salespersons[i],
        'Customer_ID': customers[i],
        'Quantity': quantity,
        'Unit_Price': round(unit_price, 2),
        'Discount_Rate': round(discount, 3),
        'Subtotal': round(subtotal, 2),
        'Discount_Amount': round(discount_amount, 2),
        'Total_Amount': round(total_amount, 2),
        'Customer_Age': np.random.randint(18, 75),
        'Customer_Satisfaction': round(np.random.uniform(2.0, 5.0), 1),
        'Payment_Method': np.random.choice(['Credit Card', 'Debit Card', 'Cash', 'Bank Transfer']),
        'Order_Priority': np.random.choice(['Low', 'Medium', 'High', 'Critical'], p=[0.4, 0.3, 0.2, 0.1])
    }
    data_records.append(record)

# Create DataFrame
df_original = pd.DataFrame(data_records)

# Add calculated columns
df_original['Month'] = df_original['Date'].dt.month
df_original['Year'] = df_original['Date'].dt.year
df_original['Quarter'] = df_original['Date'].dt.quarter
df_original['Day_of_Week'] = df_original['Date'].dt.day_name()
df_original['Month_Name'] = df_original['Date'].dt.month_name()

# Save to CSV
csv_filename = 'sales_dataset_comprehensive.csv'
df_original.to_csv(csv_filename, index=False)
print(f"Dataset saved to CSV: {csv_filename}")

# Now read from CSV to demonstrate CSV loading
print("Loading data from CSV file...")
df = pd.read_csv(csv_filename)

# Convert Date column back to datetime after reading CSV
df['Date'] = pd.to_datetime(df['Date'])

print(f"Successfully loaded CSV with shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Creating individual Pandas Series from CSV data
date_series = pd.Series(df['Date'], name='Transaction_Date')
product_series = pd.Series(df['Product'], name='Product_Name')
amount_series = pd.Series(df['Total_Amount'], name='Sale_Amount')

print(f"\nCreated Series from CSV data:")
print(f"Date Series shape: {date_series.shape}")
print(f"Product Series shape: {product_series.shape}")
print(f"Amount Series shape: {amount_series.shape}")

