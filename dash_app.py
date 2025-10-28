"""
TAKEN FROM 1
The code for creating a simple Plotly Dash app was taken from https://dash.plotly.com/tutorial (last accessed: 06.10.2025)

TAKEN FROM 2
The code for preparing the training and test sets was taken from https://medium.com/@whyamit404/understanding-train-test-split-in-pandas-eb1116576c66

TAKEN FROM 3
The solution of trying to improve the layout of a Matplotlib plot which wasn"t centered was taken from https://stackoverflow.com/a/17390833

TAKEN FROM 4
The code for using the logistic regression machine learning model was taken from https://www.digitalocean.com/community/tutorials/logistic-regression-with-scikit-learn

TAKEN FROM 5
The code for transforming non-numerical values from a Pandas dataframe into numerical values using the LabelEncoder from scikit-learn was taken from https://stackoverflow.com/a/50259157

TAKEN FROM 6
The code for changing the datatype of a value from a Pandas dataframe was taken from https://sentry.io/answers/change-a-column-type-in-a-dataframe-in-python-pandas/

TAKEN FROM 7
The code for obtaining the number of appearances of a value in a dataframe was taken from https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html

TAKEN FROM 8
The code for setting the index of a dataframe as column values was taken from https://stackoverflow.com/a/28503602

TAKEN FROM 9
The idea of using a beeswarm plot for visualizing SHAP values was taken from https://www.youtube.com/watch?v=L8_sVRhBDLU
Furthermore, the code of changing the color scheme of the plot was taken from https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/beeswarm.html

TAKEN FROM 10
The code for using saved plots in Dash apps was taken from https://community.plotly.com/t/return-render-shap-plots-in-div/38766

TAKEN FROM 11
The code for using html.Img in Dash was taken from https://community.plotly.com/t/how-to-embed-images-into-a-dash-app/61839

TAKEN FROM 12
Ideas for implementing the experimental data analysis (EDA) part were taken from: https://deepnote.com/app/code-along-tutorials/A-Beginners-Guide-to-Exploratory-Data-Analysis-with-Python-f536530d-7195-4f68-ab5b-5dca4a4c3579?utm_content=f536530d-7195-4f68-ab5b-5dca4a4c3579

TAKEN FROM 13
The idea of using the Dash component "Div" and centering everything was taken from https://stackoverflow.com/a/58089518

TAKEN FROM 14
The settings for changing the styling of the table were taken from  https://community.plotly.com/t/dash-table-change-font-and-size/20326

TAKEN FROM 15
The settings for customizing HTML headers were taken from https://dash.plotly.com/layout

TAKEN FROM 16
The settings for using a Dash Bootstrap theme were taken from https://towardsdatascience.com/3-easy-ways-to-make-your-dash-application-look-better-3e4cfefaf772/

TAKEN FROM 17
The solution for applying bold text to a HTML header was taken from https://community.plotly.com/t/how-can-i-show-a-bold-text-output/52798

TAKEN FROM 18
The code for creating a pie plot and preparing its data using Plotly was taken from https://plotly.com/python/pie-charts/

TAKEN FROM 19
The code for saving a static image using Plotly was taken from https://plotly.com/python/static-image-export/

TAKEN FROM 20
The code for summing up values form a Pandas dataframe based on a condition was taken from https://stackoverflow.com/a/28236391

TAKEN FROM 21
The code for creating a sidebar using Bootstrap for Dash apps was taken from https://www.dash-bootstrap-components.com/examples/simple-sidebar/

TAKEN FROM 22
The code for using a grid for the website layout was taken from https://www.dash-bootstrap-components.com/docs/components/layout/

TAKEN FROM 23
The code for creating a bar plot and preparing its data using Plotly was taken from https://plotly.com/python/bar-charts/

TAKEN FROM 24
The code for using Boolean conditions in single code line for pandas dataframes was taken from https://stackoverflow.com/a/27360130
"""

import random
from dash import Dash, html, dash_table, dcc, Input, Output
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

# Values which can be changed
seed = 1
random.seed(seed)
np.random.RandomState(seed)
# Test set size for the ML model
test_size = 0.2
# Maximum iterations allowed for the chosen ML model (Logistic Regression) during training
max_iter_lr = 1000

# TAKEN FROM START 1

"""
Create a table with how many bookings the travel agents made and a pie chart displaying the number of bookings made with each TA
"""

hotel_bookings_dataset = pd.read_csv("data/hotel_bookings.csv")
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
travel_agents_and_no_bookings_dataframe["Travel Agent ID"] = travel_agents_and_no_bookings_dataframe[
    "Travel Agent ID"].astype(str)
# TAKEN FROM START 18
travel_agents_and_no_bookings_dataframe.loc[
    travel_agents_and_no_bookings_dataframe["No. of bookings"] <= 10, "Travel Agent ID"] = "Other TAs combined"
# TAKEN FROM END 18
# TAKEN FROM START 20
sum_bookings_other_tas = travel_agents_and_no_bookings_dataframe.loc[
    travel_agents_and_no_bookings_dataframe["Travel Agent ID"] == "Other TAs combined", "No. of bookings"].sum()
# TAKEN FROM END 20
travel_agents_and_no_bookings_dataframe.loc[
    travel_agents_and_no_bookings_dataframe[
        "Travel Agent ID"] == "Other TAs combined", "No. of bookings"] = sum_bookings_other_tas

travel_agents_and_no_bookings_dataframe.drop_duplicates(inplace=True, keep="first")
"""
Prepare the input data for a chosen machine learning model. After training it, the SHAP values will be analyzed.
"""
# TAKEN FROM START 5
column_nonnumerical_values_to_numerical_values = ["hotel", "arrival_date_month", "meal", "country", "market_segment",
                                                  "distribution_channel",
                                                  "reserved_room_type", "assigned_room_type", "deposit_type",
                                                  "customer_type",
                                                  "reservation_status", "reservation_status_date"]
le = preprocessing.LabelEncoder()
for column in column_nonnumerical_values_to_numerical_values:
    hotel_bookings_dataset[column] = le.fit_transform(hotel_bookings_dataset[column])
# TAKEN FROM END 5

# TAKEN FROM START 2
y = hotel_bookings_dataset["is_canceled"]
X = hotel_bookings_dataset.drop("is_canceled", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
# TAKEN FROM END 2

# TAKEN FROM START 4
lr_model = LogisticRegression(max_iter=max_iter_lr, random_state=seed)
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
print("Prediction accuracy = ", accuracy)
print("Area under the ROC curve = ", roc_auc)
# TAKEN FROM END 4
assert (accuracy >= 0.5)
assert (roc_auc >= 0.5)

# TAKEN FROM START 3
plt.rcParams.update({"figure.autolayout": True})
# TAKEN FROM END 3

# TAKEN FROM START 9
model_output_explainer = shap.Explainer(lr_model, X_train)
shap_values = model_output_explainer(X_train)
shap.plots.beeswarm(shap_values, color=plt.get_cmap("cool"), show=False)
# TAKEN FROM END 9
# TAKEN FROM START 10
plt.title("Explainable AI (XAI): SHAP values")
plt.savefig("assets/shap_values_beeswarm_plot.png")
# TAKEN FROM END 10

# TAKEN FROM START 18
current_fig = px.pie(data_frame=travel_agents_and_no_bookings_dataframe,
                     values=travel_agents_and_no_bookings_dataframe["No. of bookings"],
                     names=travel_agents_and_no_bookings_dataframe["Travel Agent ID"],
                     title="Travel Agent (TA) IDs and percentages of bookings made with their help")
# TAKEN FROM END 18
# TAKEN FROM START 19
current_fig
current_fig.write_image("assets/travel_agents_and_no_bookings_pie_plot.png")
# TAKEN FROM END 19

# TAKEN FROM START 6, 7
no_bookings_per_month = hotel_bookings_dataset["arrival_date_month"].astype(int).value_counts()
# TAKEN FROM END 6, 7
# TAKEN FROM START 8
no_bookings_per_month_dataframe = pd.DataFrame(data=no_bookings_per_month).reset_index()
# TAKEN FROM END 8
no_bookings_per_month_dataframe.columns = ["Month", "No. of bookings"]
no_bookings_per_month_dataframe["Month"] = no_bookings_per_month_dataframe[
    "Month"].astype(str)
# TAKEN FROM START 24
no_bookings_per_month_dataframe = no_bookings_per_month_dataframe.drop(
    no_bookings_per_month_dataframe[no_bookings_per_month_dataframe["Month"] == "0"].index)
# TAKEN FROM END 24
# TAKEN FROM START 23
current_fig = px.bar(data_frame=no_bookings_per_month_dataframe,
                     y=no_bookings_per_month_dataframe["No. of bookings"],
                     x=no_bookings_per_month_dataframe["Month"],
                     title="Number of bookings per month")
# TAKEN FROM END 23
# TAKEN FROM START 19
current_fig.write_image("assets/no_bookings_per_month_bar_plot.png")
# TAKEN FROM END 19

# TAKEN FROM START 21
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.P("Hotel Bookings Analyzer", className="lead"),
        dbc.Nav(
            [
                dbc.NavLink("Statistics about hotel bookings", href="/", active="exact"),
                dbc.NavLink("Metrics for booking cancellation predictions", href="/metrics-prediction-cancellations",
                            active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])
# TAKEN FROM END 21

content_statistics_bookings = html.Div([
    # TAKEN FROM START 13
    # TAKEN FROM START 15
    # TAKEN FROM START 17

    html.Div(children=[html.H1(children=html.B("Statistics about hotel bookings"))],
             style=dict(display="flex", justifyContent="center")),
    # TAKEN FORM END 17
    # TAKEN FROM END 15
    # TAKEN FROM START 22
    dbc.Row([
        dbc.Col(
            html.Div(children=[
                html.Img(src="assets/no_bookings_per_month_bar_plot.png",
                         style={"height": "70%", "width": "70%"})],
                style=dict(display="flex", justifyContent="center"))),
        dbc.Col(
            html.Div(children=[
                html.Img(src="assets/travel_agents_and_no_bookings_pie_plot.png",
                         style={"height": "70%", "width": "70%"})],
                style=dict(display="flex", justifyContent="center")))
    ]
    )
    # TAKEN FROM END 22
    # TAKEN FROM END 14

])

prediction_metrics_dataframe = pd.DataFrame(
    data={"Metric": ["Prediction accuracy", "Area under the ROC curve"], "Value": [0.87, 0.6]})

content_prediction_metrics = html.Div([
    # TAKEN FROM START 13
    # TAKEN FROM START 15
    # TAKEN FROM START 17
    html.Div(children=[html.H1(children=html.B("Metrics for booking cancellation predictions"))],
             style=dict(display="flex", justifyContent="center")),
    # TAKEN FORM END 17
    # TAKEN FROM START 22
    dbc.Row([
        dbc.Col(
            html.Div(children=[
                dash_table.DataTable(data=prediction_metrics_dataframe.to_dict("records"),
                                     style_cell={"font_size": "20px",
                                                 "text_align": "center"})])),
        # TAKEN FROM END 15
        # TAKEN FROM START 11
        dbc.Col(
            html.Div(children=[html.Img(src="assets/shap_values_beeswarm_plot.png")],
                     style=dict(display="flex", justifyContent="center"))),
    ])
    # TAKEN FROM END 22
    # TAKEN FROM END 11
    # TAKEN FROM END 13
])


# TAKEN FROM START 21
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return content_statistics_bookings
    elif pathname == "/metrics-prediction-cancellations":
        return content_prediction_metrics
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )


if __name__ == "__main__":
    app.run(debug=True, port=8050)

# TAKEN FROM END 21
# TAKEN FROM END 1
