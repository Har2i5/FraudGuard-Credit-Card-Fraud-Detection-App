import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st
import zipfile

zf = zipfile.ZipFile('creditcard.zip') 
transactions = pd.read_csv(zf.open('creditcard.csv'))


# object_columns = transactions.select_dtypes(include = ['object']).columns

# for column in object_columns:
#     unique_values = transactions[column].unique()  
#     string_values = [value for value in unique_values if isinstance(value, str)]
#     print("Unique string values in column '{}':".format(column))
#     for value in string_values:
#         print(value)

transactions['V2'] = transactions['V2'].replace("I think you're doing good!", np.nan)
transactions['V2'] = transactions['V2'].astype('float64')

transactions['V7'] = transactions['V7'].replace("Hey you found me!", np.nan)
transactions['V7'] = transactions['V7'].replace("If you're good, you'll find me!", np.nan)
transactions['V7'] = transactions['V7'].astype('float64')

transactions['V9'] = transactions['V9'].replace("This is fun!", np.nan)
transactions['V9'] = transactions['V9'].astype('float64')

transactions['V24'] = transactions['V24'].replace("Nah I am caught I guess", np.nan)
transactions['V24'] = transactions['V24'].astype('float64')

transactions['V2'] = transactions['V2'].fillna(transactions['V2'].mean())
transactions['V7'] = transactions['V7'].fillna(transactions['V7'].mean())
transactions['V9'] = transactions['V9'].fillna(transactions['V9'].mean())
transactions['V24'] = transactions['V24'].fillna(transactions['V24'].mean())

# Dropping duplicates
transactions.drop_duplicates(inplace = True)



scaler = StandardScaler()
transactions[['Time','Amount']] = scaler.fit_transform(transactions[['Time','Amount']])
# Select features for PCA (standardized Time and Amount included)
pca_features = transactions[['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]]

# Initialize PCA
pca = PCA(n_components=2)

# Fitting and transforming the data
pca_result = pca.fit_transform(pca_features)

# Create a DataFrame with PCA results
pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])

# Adding Class column for visualization
pca_df['Class'] = transactions['Class'].values


pca_features = transactions[['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]]

# Initialize and fit Isolation Forest
isolation_forest = IsolationForest(contamination=0.001, random_state=101)
transactions['Anomaly_IF'] = isolation_forest.fit_predict(pca_features)

# Convert -1 (anomaly) and 1 (normal) to 1 (anomaly) and 0 (normal)
transactions['Anomaly_IF'] = np.where(transactions['Anomaly_IF'] == -1, 1, 0)

# Save the model
joblib.dump(isolation_forest, 'isolation_forest_model.pkl')



# Load your trained model
model = joblib.load('isolation_forest_model.pkl')

def detect_fraudulent_transactions(new_data, model):
    """
    Detect fraudulent transactions using a trained anomaly detection model.
    
    Parameters:
    - new_data (pd.DataFrame): A DataFrame containing the new credit card transactions.
    - model: A trained anomaly detection model (e.g., Isolation Forest).
    
    Returns:
    - pd.DataFrame: Transactions classified as fraudulent.
    """
    required_features = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
    
    if not all(feature in new_data.columns for feature in required_features):
        raise ValueError("New data must contain the following features: " + ', '.join(required_features))
    
    # Scale the new data
    scaler = StandardScaler()
    new_data[['Time', 'Amount']] = scaler.fit_transform(new_data[['Time', 'Amount']])

    # Apply the trained model to detect anomalies
    new_data['Anomaly'] = model.predict(new_data[required_features])
    
    # Convert predictions: -1 (anomaly) to 1 (fraud) and 1 (normal) to 0 (not fraud)
    new_data['Anomaly'] = np.where(new_data['Anomaly'] == -1, 1, 0)
    
    # Return the transactions classified as fraudulent
    fraudulent_transactions = new_data[new_data['Anomaly'] == 1]
    
    return fraudulent_transactions

# Streamlit app

st.title('FraudGuard: Credit Card Fraud Detection App')

st.text('FraudGuard is an intelligent app designed to enhance your financial security. By uploading your credit card transaction dataset and a trained anomaly detection model, it quickly identifies and classifies potentially fraudulent transactions. Stay one step ahead of fraud with real-time insights and alerts.')

uploaded_file = st.file_uploader("Upload your file here")

if uploaded_file is not None:
    # Load the dataset
    new_data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(new_data.head())
    
    
    # objects = new_data.select_dtypes(include = ['object']).columns

    # for column in objects:
    #     unique_values = new_data[column].unique()  
    #     string_values = [value for value in unique_values if isinstance(value, str)]
    #     print("Unique string values in column '{}':".format(column))
    #     for value in string_values:
    #         print(value)

    # Cleaning Data
    new_data['V3'] = new_data['V3'].replace("You need to clean the test data too you know", np.nan)
    new_data['V3'] = new_data['V3'].astype('float64')

    new_data['V8'] = new_data['V8'].replace("I will not leave you alone!", np.nan)
    new_data['V8'] = new_data['V8'].astype('float64')

    new_data['V16'] = new_data['V16'].replace("This as well", np.nan)
    new_data['V16'] = new_data['V16'].astype('float64')

    # Filling Missing Values
    new_data['V3'] = new_data['V3'].fillna(new_data['V3'].mean())
    new_data['V8'] = new_data['V8'].fillna(new_data['V8'].mean())
    new_data['V16'] = new_data['V16'].fillna(new_data['V16'].mean())

    
    st.header('Visualization')
    fig, ax = plt.subplots(figsize=(12,6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Class', data=pca_df, palette='coolwarm', alpha=0.6, ax = ax)
    ax.set_title('PCA of Transactions: Normal vs Fraudulent')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.legend(title='Class')
    
    st.pyplot(fig)
    st.text('The plot shows two distinct clusters: one representing normal transactions (Class 0) and another representing fraudulent transactions (Class 1).')
    

    # Run the fraud detection
    if st.button("Detect Fraudulent Transactions"):
        try:
            fraudulent_transactions = detect_fraudulent_transactions(new_data, model)
            st.write("Detected Fraudulent Transactions:")
            st.dataframe(fraudulent_transactions)
            st.write(f"Total Fraudulent Transactions Detected: {len(fraudulent_transactions)}")
        except ValueError as e:
            st.error(str(e))




