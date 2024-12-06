from prophet import Prophet
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.title("Biography")
st.write(
    "With a bachelor's degree in Biochemistry and Molecular Biology, I have cultivated a career in medical device and vaccine research, contributing to advancements in healthcare technology. Currently, I am expanding my expertise through a Master's degree in Data Science, with a focus on leveraging machine learning to optimize processes such as scheduling and data processing. My goal is to integrate data-driven solutions into my current role and collaberating with interdisciplinary teams to enhance efficiency and innovation across departments."
)

st.title("Resume")

uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file:
    st.write("**Uploader File:**", uploaded_file.name)

    if st.button("View PDF"):
        with st.expander("Click to view the PDF"):
            pdf_data = uploaded_file.read()
            st.download_button(
                "Download PDF", pdf_data, file_name=uploaded_file.name, mime="application/pdf"
            )
            st.download_button(
                f"Download: {uploaded_file.name}",
                pdf_data,
                file_name=uploaded_file.name,
            )



st.title("Overdose Death Rates")
st.write(
    "This aims to analyze historical data to predict future trends, leveraging Random Forest for feature importance and regression tasks, and Prophet for time-series forecasting"
)

uploaded_file = st.file_uploader("Upload your CSV file", type="csv", key="file_uploader_rf")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview")
    st.dataframe(df)

    # check your columns of interest
    if "ESTIMATE" not in df.columns:
        st.error("'ESTIMATE' column not found in uploaded data.")
    else:
        # check missing data for upload
        if df["ESTIMATE"].isna().sum() > 0:
            st.warning(f"Missing values detected in 'ESTIMATE'. {df['ESTIMATE'].isna().sum()} rows will be dropped.")
            df = df.dropna(subset=["ESTIMATE"])

# use the best model from checks (Random Forest)
    X = df.drop(columns=["ESTIMATE"])
    y = df["ESTIMATE"]

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # one-hot encoding features
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)

    # align the columns
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    # define the hyperarameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

# train random forest using GridSearchCV
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=5,
        verbose=0,
        n_jobs=-1
    )

    with st.spinner("Training the model. This may take some time..."):
        grid_search.fit(X_train, y_train)

    # best model and the evaluations
    best_rf_model = grid_search.best_estimator_
    st.success(f"Best Parameters: {grid_search.best_params_}")

    y_test_pred = best_rf_model.predict(X_test)
    rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    st.write(f"Test RMSE: {rmse:.2f}")

    #display feature importance
    feature_importances = best_rf_model.feature_importances_
    sorted_indices = feature_importances.argsort()[::-1]
    st.write("Top Features")
    st.write(pd.DataFrame({
        "Feature": X_train.columns[sorted_indices],
        "Importance": feature_importances[sorted_indices],
    }).head(10))

else:
    st.info("Please upload a CSV file to proceed.")

# Now we will add forecasting (choice year)
st.title("Forecast with Prophet")
uploaded_file = st.file_uploader("Upload your CSV file", type="csv", key="file_uploader_forecast")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df)

    if "YEAR" not in df.columns or "ESTIMATE" not in df.columns:
        st.error("The dataset must contain 'YEAR' and 'ESTIMATE' columns.")
    else:
        df = df.sort_values("YEAR")

        prophet_df = df.rename(columns={"YEAR": "ds", "ESTIMATE": "y"})
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"], format="%Y")

    #user can choose the forecasted year
    forecast_period = st.slider(
        "Select the forecast period (years):", min_value=1, max_value=20, value=5
    )

    # train the model
    model = Prophet()
    with st.spinner("Training the Forecasting model..."):
        model.fit(prophet_df)

    # make the future predictions
    future = model.make_future_dataframe(periods=forecast_period, freq='Y')
    forecast = model.predict(future)

    # display
    st.write("Forecast Results for the next {forecast_period} years")
    st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_period))

    # plot it so we can see more aspects
    st.write("Forecast Plot:")
    forecast_plot = model.plot(forecast)
    st.pyplot(forecast_plot)

    st.write("Forecast Components:")
    components_plot = model.plot_components(forecast)
    st.pyplot(components_plot)

else: 
    st.info("Please upload a CSV file to proceed.")