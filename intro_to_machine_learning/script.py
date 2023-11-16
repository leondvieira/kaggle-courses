import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error



# read the data and store data in DataFrame
melbourne_file_path = "./archive/melb_data.csv"
melbourne_data = pd.read_csv(melbourne_file_path)


# print a summary of the data
print(melbourne_data.describe())

# print columns
print(melbourne_data.columns)


# Filter rows with missing price values
melbourne_data = melbourne_data.dropna(axis=0)

# prediction target (by convention called as Y)
y = melbourne_data.Price

# target features (by convention called as X)
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)


###############################################################
# Fit model to predict the first 5 lines
###############################################################


# melbourne_model.fit(X, y)
# print("Making predictions for the following 5 houses:")
# print(X.head())
# print("The predictions are")
# print(melbourne_model.predict(X.head()))


###############################################################
# Fit model to predict and figure out MAE with in-sample values
###############################################################


melbourne_model.fit(X, y)
predicted_home_prices = melbourne_model.predict(X)
# print(mean_absolute_error(y, predicted_home_prices))
# 1115.7467183128902


###############################################################
# Fit model to predict and figure out MAE with training/test values
###############################################################


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

melbourne_model_train = DecisionTreeRegressor(random_state=1)
melbourne_model_train.fit(train_X, train_y)

predicted_home_prices_train = melbourne_model_train.predict(val_X)
# print(mean_absolute_error(val_y, predicted_home_prices_train))
# 273990.88056810846


###############################################################
# compare MAE with differing values of max_leaf_nodes
###############################################################


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


###############################################################
# Random Forests
###############################################################
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))


###############################################################
