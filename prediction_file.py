import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import plotly.express as px
import plotly.graph_objects as go


# sold_to_code = '9121004029'
# material = 'Regular PPC'
# county_code = 841
# sales_office = 'ZN8E'


# Funtion to predict future sales
def preditc_future_sales(sold_to_code, material, sales_office, county_code, model, label_encoder_sold_to_code, label_encoder_material, label_encoder_sales_office, df):
    # Assuming label_encoder is already fitted and used to transform the columns
    sold_to_code_encoded = label_encoder_sold_to_code.transform([sold_to_code])[0]
    material_encoded = label_encoder_material.transform([material])[0]
    sales_office_encoded = label_encoder_sales_office.transform([sales_office])[0]
    
    # Generate future dates for the next 12 months
    future_dates = pd.date_range(start=df['Date'].max() + pd.DateOffset(months=1), periods=12, freq='MS')
    
    # Create input data for prediction
    future_data = pd.DataFrame({'County Code': [county_code] * 12,
                                'Sales Office': [sales_office_encoded] * 12,
                                'Sold To Code': [sold_to_code_encoded] * 12,
                                'Material': [material_encoded] * 12,
                                'Month': future_dates.month,
                                'Year': future_dates.year,
                                'Date': future_dates})
    
    # Predict volumes for the next 12 months
    future_data['Predicted Volume'] = model.predict(future_data.drop('Date', axis=1))

    # Replace negative predicted volumes with 0
    future_data['Predicted Volume'] = future_data['Predicted Volume'].apply(lambda x: max(0, x))
    
    print(future_data)
    return [future_data, sold_to_code_encoded, material_encoded, sales_office_encoded]


# Funtion to plot graph for future sales
def plot_future_sale_graphs(predict_data, df, sold_to_code, material):

    future_data = predict_data[0]
    sold_to_code_encoded = predict_data[1]
    material_encoded = predict_data[2]
    sales_office_encoded = predict_data[3]

    # Filter historical data for the specific Sold To Code and Material
    historical_data = df[(df['Sold To Code'] == sold_to_code_encoded) & (df['Material'] == material_encoded)]

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=historical_data['Date'], y=historical_data['Volume'],
                            mode='lines+markers', name='Actual Volume',
                            marker=dict(symbol='circle')))
    fig1.add_trace(go.Scatter(x=future_data['Date'], y=future_data['Predicted Volume'],
                            mode='lines+markers', name='Future Predicted Volume',
                            marker=dict(symbol='x'), line=dict(dash='dash')))

    fig1.update_layout(
        title=f'Actual vs. Future Predicted Volumes for Sold To Code {sold_to_code} and Material {material}',
        xaxis_title='Date',
        yaxis_title='Volume',
        legend_title='Volume Type',
        template='plotly_white',
        width=1400,
        height=500
    )

    # Create figure
    fig2 = go.Figure()

    # Add traces for historical and future data
    fig2.add_trace(go.Scatter(x=historical_data['Date'], y=historical_data['Volume'],
                            mode='lines+markers', name='Actual Volume',
                            marker=dict(symbol='circle')))
    fig2.add_trace(go.Scatter(x=future_data['Date'], y=future_data['Predicted Volume'],
                            mode='lines+markers', name='Future Predicted Volume',
                            marker=dict(symbol='x'), line=dict(dash='dash')))

    # Update layout
    fig2.update_layout(
        title=f'Actual vs. Future Predicted Volumes for Sold To Code {sold_to_code} and Material {material}',
        xaxis_title='Date',
        yaxis_title='Volume',
        legend_title='Volume Type',
        template='plotly_white',
        width=1400,
        height=500
    )

    # Adjust x-axis to show each month
    fig2.update_xaxes(tickformat='%b %Y', tickmode='linear', dtick='M1', tickangle=45)

    return fig1, fig2


def predict_sales(sold_to_code, material, sales_office, county_code, model, label_encoder_sold_to_code, label_encoder_material, label_encoder_sales_office, df):
    predict_data = preditc_future_sales(sold_to_code, material, sales_office, county_code, model, label_encoder_sold_to_code, label_encoder_material, label_encoder_sales_office, df)
    figs = plot_future_sale_graphs(predict_data, df, sold_to_code, material)
    return figs