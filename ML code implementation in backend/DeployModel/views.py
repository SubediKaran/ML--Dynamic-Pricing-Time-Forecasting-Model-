from django.shortcuts import render
import pandas as pd
import numpy as np
from prophet import Prophet
from scipy import stats

def home(request):
    return render(request, "home.html")

def result(request):
    if request.method == 'GET':
        product_id = request.GET.get('product_id')

        # Load the dataset
        df = pd.read_csv('/path/to/brazilian_ecommerce_dataset.csv')

        # Filter data for the selected product_id
        product_df = df[df['product_id'] == product_id].copy()

        # Aggregate sales data by date
        product_df.loc[:, 'order_purchase_date'] = pd.to_datetime(product_df['order_purchase_timestamp']).dt.date
        sales_data = product_df.groupby(['order_purchase_date']).agg({'price': 'sum'}).reset_index()
        sales_data = sales_data.rename(columns={'order_purchase_date': 'ds', 'price': 'y'})

        # Remove outliers from sales data using Z-score
        sales_data = sales_data[(np.abs(stats.zscore(sales_data['y'])) < 1.5)]

        # Train a time series forecasting model
        model = Prophet()
        model.fit(sales_data)

        # Create a future dataframe for forecasting
        future = model.make_future_dataframe(periods=365)

        # Use the model to make price predictions for each day in the future
        forecast = model.predict(future)

        # Extract the forecasted prices
        forecasted_prices = forecast[['ds', 'yhat']].tail(365)

        # Get the dynamic price for the last day
        dynamic_price = forecasted_prices['yhat'].iloc[-1]

        return render(request, "result.html", {'dynamic_price': dynamic_price})

    return render(request, "result.html")
