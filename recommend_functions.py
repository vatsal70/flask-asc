# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
from scipy import stats
from collections import Counter
import warnings
warnings.filterwarnings("ignore")



def fetch_df(vendorCode):
    original_df = pd.read_csv('final.csv')
    original_df = original_df[original_df['Sold To Code'] == vendorCode]
    return original_df


def fetch_target_df(vendorCode):
    df_target = pd.read_csv('target-data.csv')
    df_target_vendor = df_target[df_target['Dealer Code'] == vendorCode]
    df_target_vendor.columns = df_target_vendor.columns.str.replace(' \n', ' ')
    df_target_vendor.columns = df_target_vendor.columns.str.replace('\n', ' ')
    return df_target_vendor


def load_and_preprocess_data(original_df):

    df_vendor = original_df.copy()
    df_vendor['Month'] = pd.to_datetime(df_vendor['YYYYMM'], format='%Y%m')
    df_vendor = df_vendor.groupby(['Month', 'Sold To Code', 'Sold To Name'])['Volume'].sum().reset_index()
    df_vendor.set_index('Month', inplace=True)

    df_material = original_df.copy()

    res = original_df.copy()
    res = res.groupby(['Month_Year', 'YYYYMM'])['Volume'].sum().reset_index().sort_values('YYYYMM')

    return df_vendor, df_material, res


# Calcuting the yearly sales and comparing the last 2 years sales.
def calculate_and_compare_yearly_sales(df_vendor):
    comparison = []

    yearly = df_vendor.resample('Y')['Volume'].sum()
    last_dates = df_vendor.groupby(df_vendor.index.year).apply(lambda x: x.index.max())
    new_index = pd.to_datetime([last_dates[year.year] for year in yearly.index])
    yearly.index = new_index
    yearly = yearly.sort_index()

    latest_date = yearly.index[-1]
    latest_year = latest_date.year
    latest_month = latest_date.month

    # Calculate sales for the previous two years and the current year, from January to the latest month
    years = range(latest_year - 2, latest_year + 1)
    years_sales = []
    
    for year in years:
        start_date = pd.Timestamp(year=year, month=1, day=1)
        end_date = pd.Timestamp(year=year, month=latest_month, day=1) + pd.offsets.MonthEnd(1)
        year_sales = df_vendor.loc[start_date:end_date, 'Volume'].sum()
        years_sales.append(year_sales)
    
    result_df = pd.DataFrame({'Year': years, 'Volume': years_sales})
    year1_diff = result_df.iloc[2]['Volume'] -  result_df.iloc[1]['Volume']
    year2_diff = result_df.iloc[2]['Volume'] -  result_df.iloc[0]['Volume']
    year1_pct = abs( year1_diff/result_df.iloc[1]['Volume'])*100
    year2_pct = abs( year2_diff/result_df.iloc[0]['Volume'])*100

    yearDiff_dict = {'prev_1year_diff': year1_diff, 'prev_1year_pct_diff': year1_pct, 'prev_2year_diff': year2_diff, 'prev_2year_pct_diff': year2_pct}
    

    # Compare last two years
    last_two_years = yearly.last('2Y')
    year_diff = last_two_years.iloc[-1] - last_two_years.iloc[-2]
    year_diff_pct = (year_diff / last_two_years.iloc[-2]) * 100

    comparison.append(f"Comparison of last two years:")
    comparison.append(f"Previous year sales: {last_two_years.iloc[-2]:.2f}")
    comparison.append(f"Current year sales: {last_two_years.iloc[-1]:.2f}")
    comparison.append(f"Difference: {year_diff:.2f} ({year_diff_pct:.2f}%)")
    # Compare last two years

    return result_df, yearDiff_dict, comparison


# Monthly analysis (from previous script)
def calculate_month_analysis(df_vendor):
    df_vendor['MoM_Growth'] = df_vendor['Volume'].pct_change() * 100
    significant_decrease = df_vendor[df_vendor['MoM_Growth'] < -20]
    return df_vendor


# Calcualting the maximum and minimum material sold by the vendor
def max_min_material(df1):
    pivot_table = df1.pivot_table(index='Material',  values='Volume', aggfunc='sum', fill_value=0)
    maxi = pivot_table['Volume'].idxmax()
    mini = pivot_table['Volume'].idxmin()   
    return maxi, mini


# Performing linear regression and calculating correlation between sales and time. 
def linear_regression(df_vendor):
    df_vendor['time'] = range(len(df_vendor))
    correlation = df_vendor['Volume'].corr(df_vendor['time'])
    slope, intercept, r_value, p_value, std_err = stats.linregress(df_vendor['time'], df_vendor['Volume'])
    return correlation, slope, p_value


def find_index_starting_with_value(arr, value):
    for i, element in enumerate(arr):
        if element.startswith(value):
            return element  # Return the element starts with 'value'
    return -1  # Return -1 if no element starts with 'value'


# Calculating the sales achieved for the given target
# def sales_target_calculator(df_target_vendor):
#     if df_target_vendor.empty:
#         recommend = ""
#     else:
#         fetch_target_column = find_index_starting_with_value(df_target_vendor.columns.tolist(), 'Target')
#         fetch_sales_column = find_index_starting_with_value(df_target_vendor.columns.tolist(), 'Sale')
    
#         target = df_target_vendor[fetch_target_column].iloc[0]
#         sales = df_target_vendor[fetch_sales_column].iloc[0]
#         target_date = pd.to_datetime(df_target_vendor.columns[2][-6:], format='%Y%m')
        
#         salesdone_date = df_target_vendor.columns[3][-10:]
        
#         recommend = (f"Target given for this vendor for {target_date.strftime('%B')} {target_date.year} is {target} units, till {salesdone_date} achieved {sales} units.")
        
#         diff = (sales/target)*100
#         if diff==100:
#             recommend += "Vendor has achieved his target, offer some bonus to encourage the vendor to keep up the sales. "
#         elif diff>100:
#             recommend += "Vendor has achieved his sales more than the given target, offer some bonus to encourage the vendor to keep up the sales. "
#         else:
#             recommend += f"\n* Vendor has achieved only {diff}% target, yet to achieve {100-diff}%.\n* Check if other compitators are giving the product for lesser price and reduce the price accordingly. "

#     return recommend

# Calculating the sales achieved for the given target
def sales_target_calculator(df_target_vendor):
    if df_target_vendor.empty:
        recommend = ""
    else:
        fetch_target_column = find_index_starting_with_value(df_target_vendor.columns.tolist(), 'Target')
        fetch_sales_column = find_index_starting_with_value(df_target_vendor.columns.tolist(), 'Sale')

        target = df_target_vendor[fetch_target_column].iloc[0]
        sales = df_target_vendor[fetch_sales_column].iloc[0]

        target_date = pd.to_datetime(df_target_vendor.columns[2][-6:], format='%Y%m')    
        salesdone_date = df_target_vendor.columns[3][-10:]

        recommend = (f" Target given for this vendor for {target_date.strftime('%B')} {target_date.year} is {target} units, till {salesdone_date} achieved {sales} units.")

        diff = (sales/target)*100
        if diff>=100:
            bonus_pct, bonus_units = 10, 100
            recommend += "* Vendor has achieved his target, offer some bonus to encourage the vendor to keep up the sales. "
            recommend += f"* Offer {bonus_pct}% bonus on extra sales of every {bonus_units} more units within this target month. Bonus ammount = Rs.{bonus_units*(bonus_pct/100)} for every {bonus_units} units."
        else:
            bonus_pct = 5
            recommend += f"* Vendor has achieved only {round(diff, 2)}% target, yet to achieve {100-round(diff, 2)}%."
            recommend += f"* Vendor would get Rs.{sales*10} if he sells remaining {sales} units. Offer some {bonus_pct}% bonus of Rs.{(sales*10)*(bonus_pct/100)} if the remainig {sales} units sold before the target date."
            # recommend += "\n* Check if other compitators are giving the product for lesser price and reduce the price accordingly."
    return recommend

# Generating the recommendations
def generate_recommendations(df, yearly_df, yearDiff_dict, correlation, slope, p_value, max_material, min_material, sales_target_recommendation):
    recommendations = []

    recommendations.append("Yearly recommendation:")
    month = pd.to_datetime(df.index[-1]).strftime('%B')
    recommendations.append(f"As of {month} this year, sales have reached {yearly_df.iloc[2]['Volume']} units, compared to {yearly_df.iloc[1]['Volume']} units in the same period last year.")

    if yearDiff_dict['prev_1year_diff'] < 0:
        recommendations.append(f"This represents a {yearDiff_dict['prev_1year_pct_diff']:.2f}% decrease compared to {int(yearly_df.iloc[1]['Year'])}")
        if yearDiff_dict['prev_2year_diff'] < 0:
            recommendations[-1] = (recommendations[-1] + f" and a {yearDiff_dict['prev_2year_pct_diff']:.2f}% decrease compared to {int(yearly_df.iloc[0]['Year'])} sales up to {month}.")
        else:
            recommendations[-1] = (recommendations[-1] + f" but {yearDiff_dict['prev_2year_pct_diff']:.2f}% increase compared to {int(yearly_df.iloc[0]['Year'])} sales up to {month}.")
        
    else:
        recommendations.append(f"Sales of this year is {yearDiff_dict['prev_1year_pct_diff']} ahead of last year - {yearly_df.iloc[1]['Year']} ")
        if yearDiff_dict['prev_2year_diff'] < 0:
            recommendations.append(f"but {yearDiff_dict['prev_2year_pct_diff']:.2f}% behind {int(yearly_df.iloc[0]['Year'])} sales till {month}.")
        else:
            recommendations.append(f"also {yearDiff_dict['prev_2year_pct_diff']:.2f}% ahead of {int(yearly_df.iloc[0]['Year'])} sales till {month}.")


    recommendations.append(f"*'{max_material}' is the most sold material over the past three years, while '{min_material}' is the least sold.")
    recommendations.append("Monthly recommendation:")
    if correlation < -0.5 and p_value < 0.05:
        recommendations.append("There is a significant downward trend in monthly sales over time. There may be a chance customers are moving away from products.")

    low_months = df[df['Volume'] < df['Volume'].mean() - df['Volume'].std()]
    
    if not low_months.empty:
        low_month_list = ', '.join(low_months.index.strftime('%B %Y'))
        month_year_pairs = low_month_list.split(", ")
        months = [pair.split()[0] for pair in month_year_pairs]
        month_counts = Counter(months)
        repeated_months = [month for month, count in month_counts.items() if count > 1]

        recommendations.append(f"Sales are notably low in the following months: {low_month_list}.")        

        if len(repeated_months)!=0:
            recommendations.append(f"* There is a consistent decrease in sales during {repeated_months} months from last 2 years.")
            if any(month in ['June','July','August','September','October'] for month in repeated_months): 
                recommendations.append(f"* This decline could be attributed to the rainy season affecting sale.")
            if any(month in ['March','April','May'] for month in repeated_months): 
                recommendations.append(f"* As these are the financial year-ending months, there may be a potential decrease in consumer spending as people pay taxes.")
            if any(month in ['November','December','January'] for month in repeated_months): 
                recommendations.append(f"* As these are the festival season months, sales may have decreased due to household investments. Try offering additional gifts or discounts to attract consumers.")

    if sales_target_recommendation != "":
        recommendations.append("Target recommendation:")
        for i in sales_target_recommendation.split("*"):
            recommendations.append(f"*{i}")
                
    return recommendations