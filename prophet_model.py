import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from prophet import Prophet
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import plotly.express as px
import plotly.graph_objects as go


def preprocess_csv_data(selected_dealer):

    # Load and preprocess data
    df = pd.read_csv('final.csv').fillna(0)
    # Convert YYYYMM to a datetime format representing the first day of the month
    df['Date'] = pd.to_datetime(df['YYYYMM'], format='%Y%m')

    # Drop unnecessary non-numeric columns
    df = df.drop(['YYYYMM', 'Month_Year', 'Company Code', 'State Name', 'Sold To Name'], axis=1)


    # Group by 'Sold To Code' and 'Date', then sum the volumes
    grouped_df = df.groupby([df['Date'].dt.to_period('M'), 'Sold To Code'])['Volume'].sum().reset_index()
    grouped_df.rename(columns = {'Sold To Code':'dealer', 'Date': 'ds', 'Volume':'y'}, inplace = True)

    grouped_df = grouped_df[grouped_df['dealer'] == selected_dealer]

    # Convert to timestamp format to ensure compatibility with Prophet
    grouped_df['ds'] = grouped_df['ds'].dt.to_timestamp()

    return grouped_df


def train_prophet_model(grouped_df, selected_dealer):

    # Initialize and fit the Prophet model
    model = Prophet(interval_width=0.95)
    model.fit(grouped_df)

    # Make future predictions for the next 12 months
    future = model.make_future_dataframe(periods=12, freq='MS')

    # Select a specific dealer for prediction
    # selected_dealer = 9121004016

    # Filter the future dataframe for the selected dealer
    future_dealer = future.copy()
    future_dealer['dealer'] = selected_dealer

    # Make predictions for the selected dealer only
    forecast = model.predict(future_dealer)

    # Filter historical data for the specific Sold To Code and Material
    historical_data = grouped_df[grouped_df['dealer'] == selected_dealer]


    # Create figure
    fig1 = go.Figure()

    # Add traces for historical and future data
    fig1.add_trace(go.Scatter(x=historical_data['ds'], y=historical_data['y'],
                            mode='lines+markers', name='Actual Volume',
                            marker=dict(symbol='circle')))
    fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],
                            mode='lines+markers', name='Future Predicted Volume',
                            marker=dict(symbol='x'), line=dict(dash='dash')))

    # Update layout
    fig1.update_layout(
        title=f'Actual vs. Future Predicted Volumes for Sold To Code {selected_dealer}',
        xaxis_title='Date',
        yaxis_title='Volume',
        legend_title='Volume Type',
        template='plotly_white',
        width=1400,
        height=500
    )

    # Adjust x-axis to show each month
    fig1.update_xaxes(tickformat='%b %Y', tickmode='linear', dtick='M1', tickangle=45)

    return fig1


def make_predictions(selected_dealer):
    df = preprocess_csv_data(selected_dealer)
    plot_figure = train_prophet_model(df, selected_dealer)
    return plot_figure
