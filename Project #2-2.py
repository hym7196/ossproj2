from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd


def sort_dataset(dataset_df):
    sorted_df = dataset_df.sort_values(by='year')
    return sorted_df


def split_dataset(dataset_df):
    train_df = dataset_df.iloc[:1718]
    test_df = dataset_df.iloc[1718:]

    X_train = train_df.drop(columns=['salary'])
    Y_train = train_df['salary'] * 0.001
    X_test = test_df.drop(columns=['salary'])
    Y_test = test_df['salary'] * 0.001

    return X_train, X_test, Y_train, Y_test


def extract_numerical_cols(dataset_df):
    numerical_cols = ['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP',
                      'fly', 'war']
    return dataset_df[numerical_cols]


def train_predict_decision_tree(X_train, Y_train, X_test):
    dt_model = DecisionTreeRegressor()
    dt_model.fit(X_train, Y_train)
    dt_predictions = dt_model.predict(X_test)
    return dt_predictions


def train_predict_random_forest(X_train, Y_train, X_test):
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, Y_train)
    rf_predictions = rf_model.predict(X_test)
    return rf_predictions


def train_predict_svm(X_train, Y_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm_model = SVR()
    svm_model.fit(X_train_scaled, Y_train)
    svm_predictions = svm_model.predict(X_test_scaled)
    return svm_predictions


def calculate_RMSE(labels, predictions):
    rmse = np.sqrt(mean_squared_error(labels, predictions))
    return rmse


if __name__ == '__main__':
    data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

    sorted_df = sort_dataset(data_df)
    X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)

    X_train = extract_numerical_cols(X_train)
    X_test = extract_numerical_cols(X_test)

    dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
    rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
    svm_predictions = train_predict_svm(X_train, Y_train, X_test)

    print("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))
    print("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))
    print("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))
