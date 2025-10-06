# hotel-bookings-analyzer
Hotel Bookings Analyzer - A data analysis project about hotel bookings

# Classification problem

The goal of the project is to analyze a business problem related to hotel bookings by using the CRISP-DM process.
One important classification problem to be solved is to recognize when a booking will be cancelled based on various factors. This could prevent customer churn and provide a better experience for the guests.
Therefore, a machine learning model is used for this task.

# Dataset
The used dataset for this project is "Hotel Booking Demand" [1] from Kaggle.
The dataset was originally published by a group of researchers [2], and then it was processed by people participating in the #tidytuesday project [3]. 
The data should be downloaded and placed in the "data" directory, which is inside the "src" directory ('src/data/hotel_bookings.csv').

# Technical details
- "hotel-bookings-analyzer.ipynb" contains, at the moment, the CRISP-DM steps related to data understanding and preparation, and model usage and evaluation 
- Python version: 3.12
- The needed dependencies can be found in the "requirements.txt" file, which was generated using the package manager "pip"

# References

[1] J. Mostipak (Kaggle user), "Hotel booking demand", https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand, last accessed: 06.08.2025

[2] N. Antonio, A. de Almeida, and L. Nunes, "Hotel booking demand datasets", in Data in Brief, Volume 22, p. 41-49, ISSN 2352-3409, 2019.

[3] "Hotels" dataset, #tidytuesday project, GitHub, https://github.com/rfordatascience/tidytuesday/blob/main/data/2020/2020-02-11/readme.md, last accessed: 06.08.2025