import streamlit as st
import pandas as pd
from prophet import Prophet
from io import BytesIO

def run_forecast(df, account_col, product_col, date_col, target_col, forecast_periods):
    # Ensure date is in datetime format
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['key'] = df[account_col].astype(str) + '_' + df[product_col].astype(str)

    forecast_frames = []

    for key, group in df.groupby('key'):
        prophet_df = group[[date_col, target_col]].rename(columns={date_col: 'ds', target_col: 'y'})
        if prophet_df['y'].count() < 10:
            continue

        model = Prophet(yearly_seasonality=4, seasonality_mode='multiplicative')
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=forecast_periods, freq='W')
        forecast = model.predict(future)

        forecast['key'] = key
        forecast = forecast[['ds', 'key', 'yhat', 'yearly']]
        forecast.rename(columns={'ds': date_col, 
                                 'yhat': f'{target_col} Forecast', 
                                 'yearly': 'Seasonality Indices'}, inplace=True)
        forecast_frames.append(forecast)

    if not forecast_frames:
        return df, pd.DataFrame()

    forecast_all = pd.concat(forecast_frames, ignore_index=True)
    df_merged = pd.merge(df, forecast_all, how='outer', on=['key', date_col])
    df_merged['Seasonality Indices'] = df_merged['Seasonality Indices'] + 1

    return df_merged, forecast_all

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Forecast')
    processed_data = output.getvalue()
    return processed_data

# STREAMLIT UI
st.title("📈 Zac's Prophet App")
st.markdown("Upload an Excel file to find seasonality and/or forecast time series data using Facebook's Prophet model!")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    # Get all sheet names
    excel_file = pd.ExcelFile(uploaded_file)
    sheet_names = excel_file.sheet_names

    # Let the user choose a sheet
    sheet_choice = st.selectbox("Select Sheet", sheet_names)
    df = pd.read_excel(excel_file, sheet_name=sheet_choice)
    st.success(f"Loaded sheet: {sheet_choice}")

    st.subheader("Select Columns")
    account_col = st.selectbox("Account Column", df.columns)
    product_col = st.selectbox("Product Column", df.columns)
    date_col = st.selectbox("Date Column", df.columns)
    target_col = st.selectbox("Column to Forecast", df.columns)
    #forecast_periods = st.slider("Weeks to Forecast", 0, 104)
    forecast_periods = st.number_input("Weeks to Forecast", min_value=0, value=0)


    if st.button("Run Forecast"):
        with st.spinner("Running Prophet forecasting..."):
            df_result, forecast_data = run_forecast(df, account_col, product_col, date_col, target_col, forecast_periods)

        if not forecast_data.empty:
            st.success("Forecasting complete!")
            st.dataframe(df_result.head())

            st.download_button(
                label="📥 Download Forecast as Excel",
                data=to_excel(df_result),
                file_name="forecast_output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.warning("Not enough data to forecast any groups. Please check your selections.")

