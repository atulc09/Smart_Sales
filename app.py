import pandas as pd
import joblib
import numpy as np
import streamlit as st #type: ignore
from pathlib import Path

# Load models and data
base_path = Path(__file__).resolve().parent
model_sales  = joblib.load(base_path / 'random_forest_sales_model.pkl')
model_profit = joblib.load(base_path / 'random_forest_profit_model.pkl')
scaler        = joblib.load(base_path / 'scaler.pkl')
feature_names = joblib.load(base_path / 'feature_names.pkl')

cols_to_scale = ['quantity', 'discount', 'shipping_cost', 'delivery_days']

# BUG FIX 2: Define known multi-word prefixes explicitly so split('_', 1)
# doesn't mangle 'sub_category_Bookcases' → prefix='sub', value='category_Bookcases'
KNOWN_PREFIXES = ['ship_mode', 'sub_category', 'segment', 'region', 'market', 'category']

def get_prefix_and_value(col):
    for prefix in sorted(KNOWN_PREFIXES, key=len, reverse=True):  # longest first
        if col.startswith(prefix + '_'):
            value = col[len(prefix) + 1:]
            return prefix, value
    # fallback for single-word prefixes
    if '_' in col:
        return col.split('_', 1)
    return None, None

# Build categorical groups
categorical_groups = {}
for col in feature_names:
    prefix, value = get_prefix_and_value(col)
    if prefix and value:
        categorical_groups.setdefault(prefix, []).append(value)

# Streamlit app
st.title("Sales and Profit Predictor")
st.write("Enter the features to predict sales and profit.")

# Inputs
quantity      = st.number_input("Quantity",       min_value=1,   value=1,    step=1)
discount      = st.number_input("Discount",       min_value=0.0, max_value=1.0, value=0.0, step=0.01)
shipping_cost = st.number_input("Shipping Cost",  value=0.0,     step=0.01)
delivery_days = st.number_input("Delivery Days",  min_value=1,   value=1,    step=1)
order_year    = st.number_input("Order Year",     min_value=2010, max_value=2025, value=2020, step=1)
order_month   = st.number_input("Order Month",    min_value=1,   max_value=12,   value=1,    step=1)

# BUG FIX 1: Store selections in a dict instead of relying on locals()
selections = {}
for group, options in sorted(categorical_groups.items()):
    selections[group] = st.selectbox(group.replace('_', ' ').title(), options)

if st.button("Predict"):
    # Build feature dict
    order_date_ordinal = int(pd.Timestamp(year=order_year, month=order_month, day=1).toordinal())
    data = {
        'quantity':          quantity,
        'discount':          discount,
        'shipping_cost':     shipping_cost,
        'delivery_days':     delivery_days,
        'order_year':        order_year,
        'order_month':       order_month,
        'order_date_ordinal': order_date_ordinal,
    }

    # One-hot encode categoricals using the selections dict (not locals())
    for group, options in categorical_groups.items():
        selected = selections[group]          # BUG FIX 1: use dict, not locals()
        for opt in options:
            data[f'{group}_{opt}'] = 1 if opt == selected else 0

    # Create DataFrame and enforce column order to match training
    df = pd.DataFrame([data])
    df = df[feature_names]                    # BUG FIX 3: align column order

    # Scale numeric columns
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # Predict
    pred_sales  = model_sales.predict(df)[0]
    pred_profit = model_profit.predict(df)[0]

    st.success(f"Predicted Sales:  {pred_sales:.2f}")
    st.success(f"Predicted Profit: {pred_profit:.2f}")