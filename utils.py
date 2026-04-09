import pandas as pd

def load_data():
    df = pd.read_csv("data/expenses.csv", encoding='utf-8-sig')
    df.columns = df.columns.str.strip()

    # 🔥 Handle mixed date formats safely
    df['Date'] = pd.to_datetime(df['Date'], format='mixed')

    return df

def add_transaction(date, category, amount, t_type):
    new_data = pd.DataFrame({
        'Date': [date],
        'Category': [category],
        'Amount': [amount],
        'Type': [t_type]
    })

    df = load_data()
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv('data/expenses.csv', index=False)

def get_summary(df):
    total_income = df[df['Type'] == 'Income']['Amount'].sum()
    total_expenses = df[df['Type'] == 'Expense']['Amount'].sum()
    savings = total_income - total_expenses
    return total_income, total_expenses, savings

def category_expense(df):
    return df[df['Type'] == 'Expense'].groupby('Category')['Amount'].sum()

def add_transaction(date, category, amount, t_type):
    new_data = pd.DataFrame({
        "Date": [pd.to_datetime(date)],  # 🔥 enforce format
        "Category": [category],
        "Amount": [amount],
        "Type": [t_type]
    })

    df = load_data()
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv("data/expenses.csv", index=False)

def prepare_monthly_data(df):
    df['Month'] = df['Date'].dt.to_period('M')

    monthly = df.groupby(['Month', 'Type'])['Amount'].sum().unstack().fillna(0)

    # Create new features
    monthly['Savings'] = monthly.get('Income', 0) - monthly.get('Expense', 0)

    # 🔥 NEW FEATURE
    monthly['Prev_Expense'] = monthly['Expense'].shift(1)

    monthly = monthly.dropna()  # remove first row (no previous data)

    monthly.reset_index(inplace=True)

    return monthly

from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

def train_model(monthly_df):
    X = monthly_df[['Income', 'Prev_Expense']]
    y = monthly_df['Expense']

    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)

    # Predictions on training data
    y_pred = model.predict(X)

    # Metrics
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    return model, mae, r2

def predict_next(monthly_df, model):
    last_row = monthly_df.iloc[-1]

    input_data = [[
        last_row['Income'],
        last_row['Prev_Expense']
    ]]

    prediction = model.predict(input_data)

    return round(prediction[0], 2)

def calculate_health_score(monthly_df):
    latest = monthly_df.iloc[-1]

    income = latest.get('Income', 0)
    expense = latest.get('Expense', 0)
    savings = latest.get('Savings', 0)

    if income == 0:
        return 0

    savings_ratio = savings / income

    # Score logic
    score = 0

    # Savings contribution (50 marks)
    score += min(max(savings_ratio * 100, 0), 50)

    # Expense control (30 marks)
    if expense < income * 0.7:
        score += 30
    elif expense < income:
        score += 15

    # Stability (20 marks)
    if len(monthly_df) > 1:
        prev = monthly_df.iloc[-2]['Expense']
        change = abs(expense - prev) / prev if prev != 0 else 0

        if change < 0.1:
            score += 20
        elif change < 0.3:
            score += 10

    return round(score, 2)  

def generate_recommendations(monthly_df):
    recs = []

    latest = monthly_df.iloc[-1]

    income = latest.get('Income', 0)
    expense = latest.get('Expense', 0)
    savings = latest.get('Savings', 0)

    # 💡 Savings check
    if income > 0:
        savings_ratio = savings / income

        if savings_ratio < 0.2:
            recs.append("💡 Try to save at least 20% of your income.")

    # ⚠️ Expense control
    if expense > income * 0.8:
        recs.append("⚠️ Your expenses are too high compared to income.")

    # 📉 Compare with previous month
    if len(monthly_df) > 1:
        prev_expense = monthly_df.iloc[-2]['Expense']

        if expense > prev_expense:
            recs.append("📉 Your expenses increased compared to last month.")
        elif expense < prev_expense:
            recs.append("📈 Good job! Your expenses decreased.")

    return recs   
