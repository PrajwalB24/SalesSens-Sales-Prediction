import streamlit as st
import numpy as np
import datetime as dt
import joblib
import matplotlib.pyplot as plt


current_year = dt.datetime.today().year

def predict_sales(p1, p2, p3, p4, p5, p6,p7 ):
    model = joblib.load('SalesSense_sales_pred_2_model')
    result = model.predict(np.array([[p1, p2, p3, p4, p5, p6, p7]]))
    return result

st.title("SaleSense : SALES PREDICTION")

st.sidebar.title("Input Parameters")
p1 = st.sidebar.selectbox("Item_Identifier", ['DR', 'FD', 'NC'])
p2 = st.sidebar.selectbox("Item_Type", ['Baking Goods', 'Breads', 'Breakfast', 'Canned', 'Dairy', 'Frozen Foods',' Fruits and Vegetables', 'Hard Drinks',' Health and Hygiene','Household', ' Meat', 'Others', 'Seafood', 'Snack Foods', 'Soft Drinks', 'Starchy Foods'])
p3 = st.sidebar.number_input("Item_MRP")
p4 = st.sidebar.selectbox("Outlet_Identifier", ['OUT010', 'OUT013', 'OUT017', 'OUT018', 'OUT019', 'OUT027', 'OUT035', 'OUT045', 'OUT046', 'OUT049'])
p5 = st.sidebar.selectbox("Outlet_Size", ['High', 'Medium', 'Small'])
p6 = st.sidebar.selectbox("Outlet_Type", ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'])
p7 = st.sidebar.selectbox("Outlet_Establishment_Year", list(range(1950, current_year + 1)))

if st.sidebar.button("Predict"):
    p1 = ['DR', 'FD', 'NC'].index(p1)
    p2 = ['Baking Goods', 'Breads', 'Breakfast', 'Canned', 'Dairy', 'Frozen Foods',' Fruits and Vegetables', 'Hard Drinks',' Health and Hygiene','Household', ' Meat', 'Others', 'Seafood', 'Snack Foods', 'Soft Drinks', 'Starchy Foods'].index(p2)
    p4 = ['OUT010', 'OUT013', 'OUT017', 'OUT018', 'OUT019', 'OUT027', 'OUT035', 'OUT045', 'OUT046', 'OUT049'].index(p4)
    p5 = ['High', 'Medium', 'Small'].index(p5)
    p6 = ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'].index(p6)
    p7 = current_year - p7

    result = predict_sales(p1, p2, p3, p4, p5, p6, p7)
    lower_bound = result - 712.86
    upper_bound = result + 712.86

    st.markdown(f"Predicted Value: **${result[0]:.2f}**")

    st.markdown(f"Lower Bound Value: **${lower_bound[0]:.2f}**")
    st.markdown(f"Upper Bound Value: **${upper_bound[0]:.2f}**")
    #st.markdown(f"Sales Value is between **${float(lower_bound):.2f}  and   ${float(upper_bound):.2f}**")

    #st.markdown(f"Sales Value is between **${float(lower_bound):.2f} and ${float(upper_bound):.2f}**")

    #st.write(f"Predicted Value : ", result)
    #st.write(f"Sales Value is between {lower_bound} and {upper_bound}")

    x_values = ['Lower Bound','Predicted', 'Upper Bound']
    y_values = [lower_bound, result[0],  upper_bound]
    plt.bar(x_values, y_values, color=['blue', 'green', 'red'])
    plt.xlabel('Sales')
    plt.ylabel('Value')
    plt.title('Item Outlet Sales Prediction')
    st.pyplot(plt)


