import streamlit as st
import numpy as np
import datetime as dt
import joblib
import matplotlib.pyplot as plt

current_year = dt.datetime.today().year

def predict_sales_model1(p2, p3, p4, p5, p6, p7):
    model = joblib.load('SalesSense_sales_pred_model')
    result = model.predict(np.array([[p2, p3, p4, p5, p6, p7]]))
    return result

def predict_sales_model2(p1, p2, p3, p4, p5, p6, p7):
    model = joblib.load('SalesSense_sales_pred_2_model')
    result = model.predict(np.array([[p1, p2, p3, p4, p5, p6, p7]]))
    return result

st.title("SaleSense: SALES PREDICTION")

st.sidebar.title("Input Parameters")
p1 = st.sidebar.selectbox("Item_Identifier (Model 2)", ['DR', 'FD', 'NC'])
p2 = st.sidebar.selectbox("Item_Type", ['Baking Goods', 'Breads', 'Breakfast', 'Canned', 'Dairy', 'Frozen Foods', 'Fruits and Vegetables', 'Hard Drinks', 'Health and Hygiene', 'Household', 'Meat', 'Others', 'Seafood', 'Snack Foods', 'Soft Drinks', 'Starchy Foods'])
p3 = st.sidebar.number_input("Item_MRP")
p4 = st.sidebar.selectbox("Outlet_Identifier", ['OUT010', 'OUT013', 'OUT017', 'OUT018', 'OUT019', 'OUT027', 'OUT035', 'OUT045', 'OUT046', 'OUT049'])
p5 = st.sidebar.selectbox("Outlet_Size", ['High', 'Medium', 'Small'])
p6 = st.sidebar.selectbox("Outlet_Type", ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'])
p7 = st.sidebar.selectbox("Outlet_Establishment_Year", list(range(1950, current_year + 1)))

if st.sidebar.button("Predict"):
    p1_idx = ['DR', 'FD', 'NC'].index(p1)
    p2_idx = ['Baking Goods', 'Breads', 'Breakfast', 'Canned', 'Dairy', 'Frozen Foods', 'Fruits and Vegetables', 'Hard Drinks', 'Health and Hygiene', 'Household', 'Meat', 'Others', 'Seafood', 'Snack Foods', 'Soft Drinks', 'Starchy Foods'].index(p2)
    p4_idx = ['OUT010', 'OUT013', 'OUT017', 'OUT018', 'OUT019', 'OUT027', 'OUT035', 'OUT045', 'OUT046', 'OUT049'].index(p4)
    p5_idx = ['High', 'Medium', 'Small'].index(p5)
    p6_idx = ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'].index(p6)
    p7 = current_year - p7

    result_model1 = predict_sales_model1(p2_idx, p3, p4_idx, p5_idx, p6_idx, p7)
    result_model2 = predict_sales_model2(p1_idx, p2_idx, p3, p4_idx, p5_idx, p6_idx, p7)

    lower_bound_model1 = result_model1 - 713.61  # Adjusted lower bound for Model 1
    upper_bound_model1 = result_model1 + 713.61  # Adjusted upper bound for Model 1
    lower_bound_model2 = result_model2 - 712.86
    upper_bound_model2 = result_model2 + 712.86

    st.markdown("### Model 1 Predictions (without item identifier)")
    st.markdown(f"Predicted Value: **${result_model1[0]:.2f}**")
    st.markdown(f"Lower Bound Value: **${lower_bound_model1[0]:.2f}**")
    st.markdown(f"Upper Bound Value: **${upper_bound_model1[0]:.2f}**")

    st.markdown("### Model 2 Predictions")
    st.markdown(f"Predicted Value: **${result_model2[0]:.2f}**")
    st.markdown(f"Lower Bound Value: **${lower_bound_model2[0]:.2f}**")
    st.markdown(f"Upper Bound Value: **${upper_bound_model2[0]:.2f}**")

    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    #x_values = ['Lower Bound', 'Predicted', 'Upper Bound']
    #y_values_model1 = [lower_bound_model1, result_model1[0], upper_bound_model1]
    #y_values_model2 = [lower_bound_model2, result_model2[0], upper_bound_model2]

    #ax1.bar(x_values, y_values_model1, color=['blue', 'green', 'red'])
    #ax1.set_xlabel('Sales')
    #ax1.set_ylabel('Value')
   #ax1.set_title('Model 1: Item Outlet Sales Prediction')

    #ax2.bar(x_values, y_values_model2, color=['blue', 'green', 'red'])
    #ax2.set_xlabel('Sales')
    #ax2.set_ylabel('Value')
    #ax2.set_title('Model 2: Item Outlet Sales Prediction')

    #st.pyplot(fig)



    fig, ax = plt.subplots(figsize=(8, 6))

    # Define the labels and values for the x-axis
    x_labels = ['Model 1', 'Model 2']
    x_values = np.arange(len(x_labels))

    # Predicted values and error bars for Model 1
    y_values_model1 = [result_model1[0]]
    errors_model1 = [[result_model1[0] - lower_bound_model1[0]], [upper_bound_model1[0] - result_model1[0]]]

    # Predicted values and error bars for Model 2
    y_values_model2 = [result_model2[0]]
    errors_model2 = [[result_model2[0] - lower_bound_model2[0]], [upper_bound_model2[0] - result_model2[0]]]

    

    # Plot bars for Model 1
    ax.bar(x_values[0], y_values_model1, yerr=errors_model1, width=0.4, label='Model 1', color='blue', alpha=0.7, capsize=10)

    # Plot bars for Model 2
    ax.bar(x_values[1], y_values_model2, yerr=errors_model2, width=0.4, label='Model 2', color='green', alpha=0.7, capsize=10)

    

    # Annotate the lower and upper bounds for Model 1
    ax.annotate(f'{lower_bound_model1[0]:.2f}', xy=(x_values[0], lower_bound_model1[0]), xytext=(0, -20), textcoords='offset points', ha='center', fontsize=10, color='black')
    ax.annotate(f'{upper_bound_model1[0]:.2f}', xy=(x_values[0], upper_bound_model1[0]), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=10, color='black')

    # Annotate the lower and upper bounds for Model 2
    ax.annotate(f'{lower_bound_model2[0]:.2f}', xy=(x_values[1], lower_bound_model2[0]), xytext=(0, -20), textcoords='offset points', ha='center', fontsize=10, color='black')
    ax.annotate(f'{upper_bound_model2[0]:.2f}', xy=(x_values[1], upper_bound_model2[0]), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=10, color='black')
    
    
    # Set x-axis labels and title
    ax.set_xticks(x_values)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel('Sales Value')
    ax.set_title('Comparison of Model Predictions with Error Bars')

    # Add legend
    ax.legend()

    # Show the plot
    st.pyplot(fig)