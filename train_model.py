import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import joblib



def model_train(label_encoder_sales_office, label_encoder_material, label_encoder_sold_to_code, df):
    # Prepare features and target
    X = df.drop(['Date', 'Volume'], axis=1)
    y = df['Volume']

    # Train the model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=1, max_depth=10)
    model.fit(X, y)

    # Predict volumes
    df['Predicted Volume'] = model.predict(X)

    # Plot the actual and predicted volumes
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Volume'], label='Actual Volume', marker='o')
    plt.plot(df['Date'], df['Predicted Volume'], label='Predicted Volume', marker='x')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.title('Actual vs. Predicted Volumes')
    plt.legend()
    plt.grid(True)
    plt.show()


    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(df['Volume'], df['Predicted Volume']))

    # Calculate MAE
    mae = mean_absolute_error(df['Volume'], df['Predicted Volume'])

    # Calculate R2 score
    r2 = r2_score(df['Volume'], df['Predicted Volume'])

    # Calculate MAPE
    mape = np.mean(np.abs((df['Volume'] - df['Predicted Volume']) / df['Volume'])) * 100

    print(f'RMSE: {rmse}')
    print(f'MAE: {mae}')
    print(f'R2 Score: {r2}')
    print(f'MAPE: {mape}%')

    # Save the model and encoders
    joblib.dump(model, 'FINAL/extracted/xgboost_model.pkl')
    joblib.dump(label_encoder_sales_office, 'FINAL/extracted/label_encoder_sales_office.pkl')
    joblib.dump(label_encoder_material, 'FINAL/extracted/label_encoder_material.pkl')
    joblib.dump(label_encoder_sold_to_code, 'FINAL/extracted/label_encoder_sold_to_code.pkl')
    return model