# HighRadius_Cash_Forecasting_Treasury_1st_Place_solution
This repo includes codes for Hackathon challenge held at highradius for their Treasury product Finished 1st on this competition

# Problem Statement

The goal was to build ML/DL models for cash forecasting to be used in the Treasury product. The model (s) forecasted cash received on a daily basis for 180 days into the future.

# Idea

The Idea was to create a regression but in form of a sequential model. Generally, in normal regression each row is independent to other row. But since the problem is time dependent so the approach should be time dependent as well. So, dataset needs to be preprocessed to form it in a sequential time series manner but for regression.( Basically to predict (T+1)th  record we should use T th record prediction to create features for next Row – In this way we have that sequential behavior as well as normal regression model behavior since, other features which are not dependent on previous prediction are also there in the model so the next prediction is not biased to the previous prediction completely and at the same time some information of previous prediction is there in our current row features as well).
## Following are the preprocessing steps:

1.	Firstly, we filtered the data by REVP and PMT and grouped at Effective Date to get the desired amount at day level. 

2.	This is a normal train set now we will add the dates on which no transaction was there in the system since for all those days cash received is zero hence those dates can be added with amount received to be zero. 

3.	Now, while test set we will have only 180 days (dates values) present. So, we will create a data with empty one 180 days and we need to predict for these 180 days considering them as raw input to my model.

Approach-
Now the only information one has at the time of prediction is the dates. Apart from this the other static information present with us is (holidays). And from set if we can extract some information w.r.t date level information then that can also help to predict.

Based on these key factors and trend observed while doing EDA we came up with the following features (Reason for each feature is mentioned wherever required). This is the set of features distributed in 3 models.

## Feature Engineering-

All the features are categorized in parts(type), then all the features which comes under that category will be sequenced and followed by the reason of using it. 

### Type - Rolling Mean and Variability in last 3- and 5-days w.r.t current day of week:

1.	For e.g. If today is Tuesday the value will be Mean of last 5 Tuesday’s cash received in the past dataset. (this will be day of week wise).

2.	For e.g. If today is Friday the value will be Mean of last 3 Friday’s cash received in the past dataset. (this will be day of week wise for each row).

3.	For e.g. If today is Tuesday the value will be Variance of last 5 Tuesday’s cash received in the past dataset. (this will be day of week wise).

4.	For e.g. If today is Tuesday the value will be Variance of last 3 Tuesday’s cash received in the past dataset. (this will be day of week wise).

Reason to Use (1-4): To capture the variance and trend of the amounts for last few day of week for that record.

### Type - Time shift-based features w.r.t current day of week:

5.	For e.g. If today is Tuesday what’s the value of cash received on Last Tuesday. (this is  day of week wise).

6.	For e.g. If today is Tuesday what’s the value of cash received on Last to last Tuesday. (this is day of week wise).

Reason to Use (5-6): To give the model a reference point to calculate the next corresponding day of week cash.

### Type - Peak based features (Peak w.r.t two standard deviation):

7.	No of days from last peak cash received (peak cash = median of cash in train data + 2* Standard Deviation) – If cash received is greater than this peak cash than that will be considered a peak and feature value will be how many days since last peak occurred.

Reason to Use (7): To let model know the peak trend (to capture all the highs in cash received)

### Type - Holiday based Features (Festivals and weekends):

8.	No of days from last Festival (Festival is given – so we can calculate no of days passed that day)

9.	No of days to next Festival (No of days are left in next Festival – festival data is given)

10.	No of days from last Holiday (Holiday is combination of Saturday/Sunday and festivals)

11.	No of days to next Holiday (Holiday is combination of Saturday/Sunday and festivals)

12.	Is holiday (today is holiday or not)

Reason to Use (8-12): Cash received gets affected as we get closer to or get past to the holidays in general.

### Type - Rolling Mean and Variability in last 3 to 8 days from the current day:

13.	MeanOf(T-1, T-2, T-3, T-4, T-5)th to predict Tth prediction

14.	VarianceOf(T-1, T-2, T-3, T-4, T-5)th to predict Tth prediction

Reason to Use (13-14): This is Recency factor, basically to capture recent trend generally in a short span of time the best guess is to make prediction as mean with confidence based on recent variance hence last 3-8 days trend can be very useful to help the model know the recent trend.

### Type – Static features based on Periodicity of date time

15.	day_of_week

16.	month

17.	is_weekend

18.	is_weekday

19.	is_month_start_end

Reason to Use (15-19): Since a general periodicity was there in dataset so these factors can show that trend to the model. Like November/December has more inflow than other months, or the month starting and ending has some different behavior, or on weekends there are basically very less transactions (this can be handled with heuristics)

For Plots of EDA you can see this link.

### Test Set Feature Calculations:

For this purpose, to create features we predict one prediction at a time and then that prediction was used to calculate last 5 days mean/variance or no of days from last peak. Hence, in this way last day prediction was used to predict for next day and so on.
Modelling:

3 XGBOOST models were used with different feature sets based on max ensemble.
