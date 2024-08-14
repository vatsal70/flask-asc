import os
import joblib
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go


from scipy import stats
from flask_cors import CORS
from itertools import product
from plotly.offline import plot
from collections import Counter
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from flask import Flask, render_template, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from apscheduler.schedulers.background import BackgroundScheduler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from train_model import model_train
from process_df import preprocess_data
from prediction_file import predict_sales
from prophet_model import make_predictions
from helper_functions import getAllProductsDataframe, getAllOrdersDataframe, createMatrixTables, recommend_products
from recommend_functions import fetch_target_df, load_and_preprocess_data, calculate_and_compare_yearly_sales, calculate_month_analysis, max_min_material, linear_regression, sales_target_calculator, generate_recommendations



warnings.filterwarnings('ignore')

app = Flask(__name__)

# List of allowed origins
allowed_origins = ["http://localhost:19006", "https://localhost:19006", "http://aioms.azurewebsites.net", "https://aioms.azurewebsites.net", "http://ascendumindia.in", "https://ascendumindia.in"]

# Enable CORS for the specified origins
CORS(app, resources={r"/*": {"origins": allowed_origins}})

app.config["MONGO_URI"] = "mongodb+srv://piwesek966:KPyxti4tbgX5TPgR@ascendumcms.chijt97.mongodb.net/cms_ecom?retryWrites=true&w=majority&appName=AscendumCMS"
mongo = PyMongo(app)

# Initialize the scheduler
scheduler = BackgroundScheduler()


file_paths = ['extracted/xgboost_model.pkl', 'extracted/label_encoder_sales_office.pkl', 'extracted/label_encoder_material.pkl', 'extracted/label_encoder_sold_to_code.pkl']

file_exists = False
for item in file_paths:
    if os.path.exists(item):
        print(f"File {item} exists.")
        file_exists = True
    else:
        print("File does not exist.")
        file_exists = False

label_encoder_sales_office, label_encoder_material, label_encoder_sold_to_code, df, original_df = preprocess_data()
model = joblib.load('extracted/xgboost_model.pkl')

@app.route('/')
def index():
    data_frame = pd.read_csv('final.csv').fillna(0)
    dealer_codes = data_frame['Sold To Code'].unique()
    return render_template('index.html', dealer_codes=dealer_codes)


@app.route('/model/<int:code>')
def model_type(code):
    return render_template('models.html', code=code)

import plotly.io as pio
@app.route('/dealer/xg/<int:sold_to_code>')
def dealer_graphs_xgboost(sold_to_code):
    material = 'Regular PPC'
    county_code = 841
    sales_office = 'ZN8E'

    fig1, fig2 = predict_sales(sold_to_code, material, sales_office, county_code, model, label_encoder_sold_to_code, label_encoder_material, label_encoder_sales_office, df)

    # Convert figures to JSON
    fig1_json = pio.to_json(fig1)
    fig2_json = pio.to_json(fig2)

    # Prepare JSON response
    response = {
        'FIGURE_ONE': fig1_json,
        'FIGURE_TWO': fig2_json
    }

    return jsonify(response)
    # plot_divs = {
    #     'FIGURE_ONE': plot(fig1, output_type='div', include_plotlyjs=False),
    #     'FIGURE_TWO': plot(fig2, output_type='div', include_plotlyjs=False),
    # }

    # return render_template('dealer_graphs_xg.html', plot_divs=plot_divs)


@app.route('/dealer/prophet/<int:sold_to_code>')
def dealer_graphs_prophet(sold_to_code):
    material = 'Regular PPC'
    county_code = 841
    sales_office = 'ZN8E'

    forecast_prophet = make_predictions(sold_to_code)

    # Convert figures to JSON
    fig1_json = pio.to_json(forecast_prophet)

    # Prepare JSON response
    response = {
        'FIGURE_ONE': fig1_json,
    }

    return jsonify(response)

    # plot_divs = {
    #     'FIGURE_ONE': plot(forecast_prophet, output_type='div', include_plotlyjs=False),
    # }

    # return render_template('dealer_graphs_prophet.html', plot_divs=plot_divs)

# Global variables
product_dataframe = None
order_dataframe = None
interaction_matrix = None
item_similarity_df = None
order_df_aggregated = None


def initialize_data():
    global product_dataframe, order_dataframe, interaction_matrix, item_similarity_df, order_df_aggregated
    
    # Products dataframe
    product_dataframe = getAllProductsDataframe(mongo)

    # Orders dataframe
    order_dataframe = getAllOrdersDataframe(mongo)

    # Interaction Matrix and Item Similarity dataframe
    interaction_matrix, item_similarity_df = createMatrixTables(order_dataframe)

    # Aggregate duplicate entries by counting the number of purchases
    order_df_aggregated = order_dataframe.groupby(
        ['user_id', 'product_id', 'product_name', 'category', 'sub_category']
    ).agg({'quantity': 'sum'}).reset_index()

    print("DATA has been initialized.")

# Function to be scheduled
def scheduled_task():
    initialize_data()
    print("This is a scheduled task to fetch the latest data")

scheduler.add_job(scheduled_task, 'cron', hour=11, minute=11)

@app.route("/recommend/products/<int:user_id>", methods=["GET"])
def mongo_page(user_id):
    try:

        # Combine the purchased products with the product inventory
        merged_products = pd.merge(product_dataframe, order_df_aggregated, how='left', on='product_id')
        merged_products['quantity'] = merged_products['quantity'].fillna(0)

        # One-hot encode the categories and sub_categories
        encoder = OneHotEncoder()
        category_encoded = encoder.fit_transform(product_dataframe[['category', 'sub_category']])

        # Calculate the cosine similarity
        cosine_sim = cosine_similarity(category_encoded)

        # Convert to DataFrame for easier handling
        cosine_sim_df = pd.DataFrame(cosine_sim, index=product_dataframe.index, columns=product_dataframe.index)

        # Recommend products for user with id '64'
        recommended_products = recommend_products(str(user_id), interaction_matrix, item_similarity_df, product_dataframe)
        print(f"\nRecommended Products for user {user_id}:")
        resp = {
            "status": "success",
            "products": recommended_products
        }
        return jsonify(resp)
    except Exception as e:
        print("ERRORRRRRRRRRRRRRRRR", e)


@app.route('/dealer/graphs/<int:sold_to_code>')
def plot_graphs(sold_to_code):
    df_Vendor = original_df[original_df['Sold To Code']==sold_to_code]
    vendorName = df_Vendor['Sold To Name'].values[0]
    df_Vendor['Date'] = pd.to_datetime(df_Vendor['YYYYMM'], format='%Y%m')

    # Group by 'Date' and sum the 'Volume' column
    df_monthly_sales = df_Vendor.groupby('Date')['Volume'].sum().reset_index()

    # Rename columns for clarity
    df_monthly_sales.columns = ['Date', 'Total Sales']

    sales_analysis = generate_sales_analysis(df_Vendor, vendorName)
    yearly_material_sales = generate_yearly_material_sales(df_Vendor, vendorName)
    forecast_graph = generate_forecast_graph(df_Vendor, vendorName)
    mom_growth_graph = generate_growth_graph(df_monthly_sales, 'MoM', vendorName)
    yoy_growth_graph = generate_growth_graph(df_monthly_sales, 'YoY', vendorName)
    sales_growth_rate_graph = generate_sales_growth_rate_graph(df_monthly_sales, vendorName)

    # Convert figures to JSON
    sales_analysis_json = pio.to_json(sales_analysis)
    yearly_material_sales_json = pio.to_json(yearly_material_sales)
    forecast_graph_json = pio.to_json(forecast_graph)
    mom_growth_graph_json = pio.to_json(mom_growth_graph)
    yoy_growth_graph_json = pio.to_json(yoy_growth_graph)
    sales_growth_rate_graph_json = pio.to_json(sales_growth_rate_graph)

    # Prepare JSON response
    response = {
        'sales_analysis_json': sales_analysis_json,
        'yearly_material_sales_json': yearly_material_sales_json,
        'forecast_graph_json': forecast_graph_json,
        'mom_growth_graph_json': mom_growth_graph_json,
        'yoy_growth_graph_json': yoy_growth_graph_json,
        'sales_growth_rate_graph_json': sales_growth_rate_graph_json
    }

    return jsonify(response)

    # plot_divs = {
    #     'sales_analysis': plot(sales_analysis, output_type='div', include_plotlyjs=False),
    #     'yearly_material_sales': plot(yearly_material_sales, output_type='div', include_plotlyjs=False),
    #     'forecast_graph': plot(forecast_graph, output_type='div', include_plotlyjs=False),
    #     'mom_growth_graph': plot(mom_growth_graph, output_type='div', include_plotlyjs=False),
    #     'yoy_growth_graph': plot(yoy_growth_graph, output_type='div', include_plotlyjs=False),
    #     'sales_growth_rate_graph': plot(sales_growth_rate_graph, output_type='div', include_plotlyjs=False),

    # }
 
    # return render_template('sales_analysis.html', plot_divs=plot_divs)



@app.route('/dealer/recommend-comparison/<int:sold_to_code>')
def recommendation_comparison(sold_to_code):

    df_Vendor = original_df[original_df['Sold To Code']==sold_to_code]
    vendorName = df_Vendor['Sold To Name'].values[0]

    df_target = fetch_target_df(sold_to_code)
    df, df1, res = load_and_preprocess_data(df_Vendor)
    yearly_df, yearDiff_dict, comparison = calculate_and_compare_yearly_sales(df)
    df = calculate_month_analysis(df)
    max_material, min_material = max_min_material(df1)
    correlation, slope, p_value = linear_regression(df)
    sales_target_recommendation = sales_target_calculator(df_target)
    recommendations = generate_recommendations(df, yearly_df, yearDiff_dict, correlation, slope, p_value, max_material, min_material, sales_target_recommendation)


    params = {
        'vendorName': vendorName,
        'recommendations': recommendations,
        'comparison': comparison
    }
    return jsonify(params)



def generate_sales_analysis(df_Vendor, vendorName):
    
    # Calculating total sales for each Month of a Year
    res = df_Vendor.groupby(['Date'])['Volume'].sum().reset_index().sort_values('Date')

    # Calculating monthly sales for each material
    df_material_sales = df_Vendor.pivot_table(index='Date', columns='Material', values='Volume', aggfunc='sum').fillna(0)

    # Reset the index to have 'Date' as a column for Plotly Express
    df_material_sales = df_material_sales.reset_index()

    # Melt the DataFrame to long format suitable for Plotly Express
    df_melted = df_material_sales.melt(id_vars=['Date'], var_name='Material', value_name='Volume')

    # Create the line plot using Plotly Express
    fig = px.line(df_melted, x='Date', y='Volume', color='Material', markers=True,
                title=f'Monthly Sales Volume of Each Material for {vendorName}')

    # Update layout for better readability
    fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Volume',
    xaxis_tickformat='%b %Y',  # Format the date as month-year
    legend_title_text='Material',
    width=1400,
    height=500,
    xaxis=dict(
            tickformat='%b %Y',
            tickmode='linear',
            dtick='M1',
            tickangle=45,
            range=[df_Vendor['Date'].min(), df_Vendor['Date'].max()]  # Setting the range to cover the full data range
        )
    )

    # Adjust x-axis to show each month
    fig.update_xaxes(tickformat='%b %Y', tickmode='linear', dtick='M1', tickangle=45)

    return fig


def generate_yearly_material_sales(df_Vendor, vendorName):
    # Calculate the yearly sales for each material
    df_Vendor['Year'] = df_Vendor['YYYYMM'].astype(str).str[:4]
    yearly_sales = df_Vendor.groupby(['Year', 'Material'])['Volume'].sum().unstack().fillna(0)

    # Reset the index and melt the DataFrame to a long format suitable for Plotly Express
    yearly_sales = yearly_sales.reset_index()
    df_melted = yearly_sales.melt(id_vars=['Year'], var_name='Material', value_name='Volume')

    # Create the line plot using Plotly Express
    fig = px.line(df_melted, x='Year', y='Volume', color='Material', markers=True,
                title=f'Yearly Sales Volume of Each Material for {vendorName}')

    # Update layout for better readability
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Volume',
        width=1400,
        height=500,
        xaxis=dict(
                tickformat='%b %Y',
                tickmode='linear',
                dtick='M1',
                tickangle=45,
                range=[df_Vendor['Date'].min(), df_Vendor['Date'].max()]  # Setting the range to cover the full data range
            )
    )

    return fig
 
 
def generate_forecast_graph(df_Vendor, vendorName):

    # Calculating total sales for each Month of a Year
    res = df_Vendor.groupby(['Month_Year', 'YYYYMM'])['Volume'].sum().reset_index().sort_values('YYYYMM')

    # Convert 'YYYYMM' to datetime format for time series analysis
    res['Date'] = pd.to_datetime(res['YYYYMM'], format='%Y%m')

    # Set the date as the index
    res.set_index('Date', inplace=True)

    # Drop the 'YYYYMM' and 'Month_Year' columns as they are no longer needed
    res.drop(columns=['YYYYMM', 'Month_Year'], inplace=True)

    # Rename 'Volume ' to 'Volume' for simplicity
    res.rename(columns={'Volume ': 'Volume'}, inplace=True)

    # # Split the data into training and test sets
    # # train_size = int(len(res) * 0.8)
    # train_size = '2023-05-01'
    # train, test = res[:train_size], res[train_size:]

    # # Fit the SARIMA model on the training data
    # model = SARIMAX(train['Volume'], order=(0, 1, 1), seasonal_order=(0, 1, 1, 12))
    # results = model.fit()

    # # Forecast for the test period
    # n_test_periods = len(test)
    # forecast = results.get_forecast(steps=n_test_periods)
    # forecast_df = forecast.conf_int(alpha=0.05)
    # forecast_df['Forecast'] = forecast.predicted_mean
    # forecast_df.index = test.index


    # # Assuming train, test, and forecast_df are pandas DataFrames with a DateTime index
    # train = pd.DataFrame({'Volume': [100, 200, 300]}, index=pd.date_range('2021-01-01', periods=3, freq='M'))
    # test = pd.DataFrame({'Volume': [150, 250, 350]}, index=pd.date_range('2021-04-01', periods=3, freq='M'))
    # forecast_df = pd.DataFrame({'Forecast': [160, 260, 360]}, index=pd.date_range('2021-04-01', periods=3, freq='M'))

    # Fit the SARIMA model on the entire dataset and forecast the next 12 months
    model = SARIMAX(res['Volume'], order=(0, 1, 1), seasonal_order=(0, 1, 1, 12))
    results = model.fit()

    # Forecast for the next 12 months
    forecast_next_12_months = results.get_forecast(steps=12)
    forecast_index = pd.date_range(start=res.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')
    forecast_next_12_df = forecast_next_12_months.conf_int(alpha=0.05)
    forecast_next_12_df['Forecast'] = forecast_next_12_months.predicted_mean
    forecast_next_12_df.index = forecast_index
    
    # Create the plotly figure
    fig = go.Figure()

    # Add traces for historical sales and forecasted sales
    fig.add_trace(go.Scatter(x=res.index, y=res['Volume'], name='Historical Sales', mode='lines+markers', marker=dict(symbol='circle')))
    fig.add_trace(go.Scatter(x=forecast_next_12_df.index, y=forecast_next_12_df['Forecast'], name='Forecasted Sales', line=dict(color='red'), mode='lines+markers', marker=dict(symbol='x')))

    # Update layout to match the Matplotlib plot
    fig.update_layout(
        width=1400,
        height=500,
        title='Sales Forecast for Next 12 Months',
        xaxis_title='Date',
        yaxis_title='Volume',
        legend_title='Legend',
        xaxis=dict(
            tickformat='%Y-%m',
            tickmode='auto',
            nticks=len(pd.date_range(start=res.index.min(), end=forecast_next_12_df.index.max(), freq='MS'))
        ),
        xaxis_tickangle=-45
    )
    return fig
 

def generate_growth_graph2(dealer_data, growth_type, vendorName):
    if growth_type == 'MoM':
        dealer_data['Growth'] = dealer_data['Total Sales'].pct_change() * 100
    elif growth_type == 'YoY':
        dealer_data['Growth'] = dealer_data['Total Sales'].pct_change(periods=12) * 100
    
    # Filter out rows where 'Growth' is NaN
    filtered_data = dealer_data.dropna(subset=['Growth'])

    fig = px.line(filtered_data, x='Date', y='Growth', title=f'{growth_type} Growth Rate for {vendorName}', markers=True)
    # Update layout for better readability
    fig.update_layout(
    width=1400,
    height=500,
    xaxis=dict(
            tickformat='%b %Y',
            tickmode='linear',
            dtick='M1',
            tickangle=45,
            range=[filtered_data['Date'].min(), filtered_data['Date'].max()]  # Setting the range to cover the full data range
        )
    )
    fig.update_xaxes(tickformat='%b %Y', tickmode='linear', dtick='M1', tickangle=45)
    return fig


def generate_growth_graph(dealer_data, growth_type, vendorName):
    if growth_type == 'MoM':
        dealer_data['Growth'] = dealer_data['Total Sales'].pct_change() * 100
    elif growth_type == 'YoY':
        dealer_data['Growth'] = dealer_data['Total Sales'].pct_change(periods=12) * 100
    fig = px.line(dealer_data, x='Date', y='Growth', title=f'{growth_type} Growth Rate for {vendorName}', markers=True)

    fig.update_layout(
    width=1400,
    height=500,
    )
    return fig
 

def generate_quartile_analysis_graph(all_sales_data, vendorName):
    fig = px.box(all_sales_data, y='Total Sales', title=f'Quartile Analysis of Sales for {vendorName}')
    # Update layout for better readability
    fig.update_layout(
    width=1400,
    height=500,
    xaxis=dict(
            tickformat='%b %Y',
            tickmode='linear',
            dtick='M1',
            tickangle=45,
            range=[all_sales_data['Date'].min(), all_sales_data['Date'].max()]  # Setting the range to cover the full data range
        )
    )
    fig.update_xaxes(tickformat='%b %Y', tickmode='linear', dtick='M1', tickangle=45)
    return fig


def generate_sales_growth_rate_graph(dealer_data, vendorName):
    dealer_data['Moving Average'] = dealer_data['Total Sales'].rolling(window=3).mean()
    fig = px.line(dealer_data, x='Date', y='Moving Average', title=f'Sales Growth Rate Over Time for {vendorName}', markers=True)
    # Update layout for better readability
    fig.update_layout(
    width=1400,
    height=500,
    xaxis=dict(
            tickformat='%b %Y',
            tickmode='linear',
            dtick='M1',
            tickangle=45,
            range=[dealer_data['Date'].min(), dealer_data['Date'].max()]  # Setting the range to cover the full data range
        )
    )
    fig.update_xaxes(tickformat='%b %Y', tickmode='linear', dtick='M1', tickangle=45)
    return fig


if __name__ == "__main__":
    # Initialize data before the first request
    initialize_data()
    
    # Start the scheduler before the application runs
    scheduler.start()
    print("Scheduler started.")

    try:
        #app.run(host='0.0.0.0', port=5000, debug=True, ssl_context = ("cert.pem", "key.pem"))
        app.run(host='0.0.0.0', port=5000, debug=True)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        # Shut down the scheduler when exiting the app
        scheduler.shutdown()
        print("Scheduler shut down.")
