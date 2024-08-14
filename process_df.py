import pandas as pd
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.width',500)
pd.set_option('display.float_format',lambda x :'%.3f' % x)


def preprocess_data():

    # Load and preprocess data
    df = pd.read_csv('final.csv').fillna(0)

    original_df = df.copy()

    # Convert YYYYMM to a datetime format representing the first day of the month
    df['Date'] = pd.to_datetime(df['YYYYMM'], format='%Y%m')

    # Drop unnecessary non-numeric columns
    df = df.drop(['YYYYMM', 'Month_Year', 'Company Code', 'State Name', 'Sold To Name'], axis=1)

    # Encode categorical columns
    label_encoder_sales_office = LabelEncoder()
    label_encoder_material = LabelEncoder()
    label_encoder_sold_to_code = LabelEncoder()

    df['Sales Office'] = label_encoder_sales_office.fit_transform(df['Sales Office'])
    df['Material'] = label_encoder_material.fit_transform(df['Material'])
    df['Sold To Code'] = label_encoder_sold_to_code.fit_transform(df['Sold To Code'])

    # Add additional features
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    df.to_csv('encoded-data.csv', index=False)

    return label_encoder_sales_office, label_encoder_material, label_encoder_sold_to_code, df, original_df