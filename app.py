"""
TAKEN FROM 1
The code for creating a simple Plotly Dash app was taken from
https://dash.plotly.com/tutorial (last accessed: 06.10.2025)

TAKEN FROM 2
The code for preparing the training and test sets was taken from https://medium.com/@whyamit404/understanding-train-test-split-in-pandas-eb1116576c66

TAKEN FROM 3
The solution of trying to improve the layout of a Matplotlib plot which wasn't centered was taken from https://stackoverflow.com/a/17390833

TAKEN FROM 4
The code for using the logistic regression machine learning model was taken from https://www.digitalocean.com/community/tutorials/logistic-regression-with-scikit-learn

TAKEN FROM 5
The code for transforming non-numerical values from a pandas dataframe into numerical values using the LabelEncoder from scikit-learn was taken from https://stackoverflow.com/a/50259157

TAKEN FROM 6
The code for converting a numerical value to an integer value was taken from https://sentry.io/answers/change-a-column-type-in-a-dataframe-in-python-pandas/

TAKEN FROM 7
The code for obtaining the number of appearances of a value in a dataframe was taken from https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html

TAKEN FROM 8
The code for setting the index of a dataframe as column values was taken from https://stackoverflow.com/a/28503602

TAKEN FROM 9
The idea of using a beeswarm plot for visualizing Shapley values was taken from https://www.youtube.com/watch?v=L8_sVRhBDLU
Furthermore, the code of changing the color scheme of the plot was taken from https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/beeswarm.html

TAKEN FROM 10
The code for using saved plots in Dash apps was taken from https://community.plotly.com/t/return-render-shap-plots-in-div/38766

TAKEN FROM 11
The code for using html.Img in Dash was taken from https://community.plotly.com/t/how-to-embed-images-into-a-dash-app/61839

TAKEN FROM 12
Ideas for implementing the experimental data analysis (EDA) part were taken from: https://deepnote.com/app/code-along-tutorials/A-Beginners-Guide-to-Exploratory-Data-Analysis-with-Python-f536530d-7195-4f68-ab5b-5dca4a4c3579?utm_content=f536530d-7195-4f68-ab5b-5dca4a4c3579
"""

import random
from dash import Dash, html, dash_table
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

seed = 1
random.seed(seed)
test_size = 0.2
max_iter_lr = 1000

# TAKEN FROM START 1

"""
Create a table with how many bookings each travel agent made.
"""

hotel_bookings_dataset = pd.read_csv('data/hotel_bookings.csv')
# TAKEN FROM START 12
hotel_bookings_dataset = hotel_bookings_dataset.dropna()
# TAKEN FROM END 12

# TAKEN FROM START 6, 7
travel_agents_and_no_bookings = hotel_bookings_dataset["agent"].astype(int).value_counts()
# TAKEN FROM END 6, 7
# TAKEN FROM START 8
travel_agents_and_no_bookings_dataframe = pd.DataFrame(data=travel_agents_and_no_bookings).reset_index()
# TAKEN FROM END 8
travel_agents_and_no_bookings_dataframe.columns = ["Travel Agent ID", "No. of bookings"]

"""
Prepare the input data for a chosen machine learning model. After training it, the Shapley values will be analyzed.
"""
# TAKEN FROM START 5
column_nonnumerical_values_to_numerical_values = ['hotel', 'arrival_date_month', 'meal', 'country', 'market_segment',
                                                  'distribution_channel',
                                                  'reserved_room_type', 'assigned_room_type', 'deposit_type',
                                                  'customer_type',
                                                  'reservation_status', 'reservation_status_date']
le = preprocessing.LabelEncoder()
for column in column_nonnumerical_values_to_numerical_values:
    hotel_bookings_dataset[column] = le.fit_transform(hotel_bookings_dataset[column])
# TAKEN FROM END 5

# TAKEN FROM START 2
y = hotel_bookings_dataset['is_canceled']
X = hotel_bookings_dataset.drop('is_canceled', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
# TAKEN FROM END 2

# TAKEN FROM START 4
lr_model = LogisticRegression(max_iter=max_iter_lr)
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
assert (accuracy >= 50)
assert (roc_auc >= 50)
# TAKEN FROM END 4

# TAKEN FROM START 3
plt.rcParams.update({'figure.autolayout': True})
# TAKEN FROM END 3

# TAKEN FROM START 9
explainer = shap.Explainer(lr_model, X_train)
shap_values = explainer(X_train)
shap.plots.beeswarm(shap_values, color=plt.get_cmap("cool"), show=False)
# TAKEN FROM END 9
# TAKEN FROM START 10
plt.savefig("assets/shap_values_beeswarm_plot.png")
# TAKEN FROM END 10

app = Dash()
app.layout = [
    html.Div(children="Travel agencies and the number of bookings that each of them made:"),
    dash_table.DataTable(data=travel_agents_and_no_bookings_dataframe.to_dict("records"), page_size=10),
    html.Div(children="XAI: Shapley values of features of the dataset, while taking into account the training set:"),
    # TAKEN FROM START 11
    html.Img(src="assets/shap_values_beeswarm_plot.png")
    # TAKEN FROM END 11
]

if __name__ == '__main__':
    app.run(debug=True)
# TAKEN FROM END 1
