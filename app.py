import streamlit as st

from utils import load_data, add_transaction, get_summary, category_expense, prepare_monthly_data, train_model, predict_next
from utils import calculate_health_score
from utils import generate_recommendations
import pandas as pd
from utils import load_data, add_transaction, get_summary, category_expense

st.set_page_config(page_title="AI Finance Advisor", layout="wide")

st.title("💰 AI Personal Finance Advisor")

# Load data
df = load_data()

# ---- Sidebar Filters ----
st.sidebar.header("Filters")

df['Month'] = df['Date'].dt.to_period('M')
selected_month = st.sidebar.selectbox("Select Month", df['Month'].astype(str).unique())

filtered_df = df[df['Month'].astype(str) == selected_month]

# ---- Add Transaction ----
st.sidebar.header("Add Transaction")

date = st.sidebar.date_input("Date")
category = st.sidebar.text_input("Category")
amount = st.sidebar.number_input("Amount")
t_type = st.sidebar.selectbox("Type", ["Expense", "Income"])

if st.sidebar.button("Add Transaction"):
    add_transaction(date, category, amount, t_type)
    st.sidebar.success("Added!")
    st.rerun()

# ---- Summary Cards ----
income, expense, savings = get_summary(filtered_df)

col1, col2, col3 = st.columns(3)

col1.metric("💰 Income", f"₹{income}")
col2.metric("💸 Expense", f"₹{expense}")
col3.metric("📈 Savings", f"₹{savings}")

# ---- Charts ----
col4, col5 = st.columns(2)

with col4:
    st.subheader("Expenses by Category (Bar)")
    cat_data = category_expense(filtered_df)
    st.bar_chart(cat_data)

with col5:
    st.subheader("Expenses Distribution (Pie)")
    if not cat_data.empty:
        pie_df = cat_data.reset_index()
        st.pyplot(pie_df.set_index('Category').plot.pie(
            y='Amount',
            autopct='%1.1f%%',
            figsize=(5,5),
            legend=False
        ).figure)

# ---- Insights (Simple AI-like logic) ----
st.subheader("💡 Smart Insight")

if not cat_data.empty:
    top_category = cat_data.idxmax()
    top_value = cat_data.max()

    st.info(f"You are spending the most on **{top_category} (₹{top_value})**")

    if savings < 0:
        st.error("⚠️ You are overspending! Reduce expenses.")
    elif savings < income * 0.2:
        st.warning("⚠️ Your savings are low. Try to save at least 20% of income.")
    else:
        st.success("✅ Great! You are saving well.")

st.subheader("🤖 AI Prediction")

monthly_df = prepare_monthly_data(df)

if len(monthly_df) >= 2:
    model = train_model(monthly_df)
    prediction = predict_next(monthly_df, model)

    st.success(f"Predicted next month expense: ₹{prediction}")
else:
    st.warning("Not enough data for prediction.")   


st.subheader("💯 Financial Health Score")
score = calculate_health_score(monthly_df)
st.metric("Your Score", f"{score}/100")
if score < 40:
    st.error("⚠️ Poor financial health. You need to control expenses.")
elif score < 70:
    st.warning("⚠️ Average financial health. Try improving savings.")
else:
    st.success("✅ Great financial health! Keep it up.")

st.subheader("💡 Smart Recommendations")

recs = generate_recommendations(monthly_df)

if recs:
    for r in recs:
        st.write(r)
else:
    st.success("✅ Your finances look well balanced!")

# ---- Raw Data ----
st.subheader("📄 Transactions")
st.dataframe(filtered_df.sort_values(by="Date", ascending=False))




