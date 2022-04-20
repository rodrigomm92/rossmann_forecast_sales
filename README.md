# rossmann_forecast_sales
This is a Forecast Project for my portfolio. In this project we are interested in the sales prediction of every Rossmann store, in order to allocate the right amount of investment to each store according to its expected income. To solve this problem we will build a regression machine learning model to predict the sales of the next 6 weeks.

<img src="banner.jpg" alt="logo" style="zoom:9% ;" />


# 1.0 - Business Problem (Fictitious Scenario)

After a good season of sales, the Rossmann company wants to invest part of its previous profit into the stores all around the country. However, the CEO wants to allocate the right amount of money for each store depending on its future revenue. After a meeting with the financial, business and data team, a strategy was set: Predict the sales of the future 6 weeks for every Rossmann store. To solve this problem, the data team was told to develop a model to make these predictions. In this way, the other teams can set the best strategy for this investment. In my <a href="https://github.com/rodrigomm92/rossmann_forecast_sales/blob/main/rossmann_forecast.ipynb">Jupyter Notebook</a> you can follow all steps of this task.

When other strategic managers heard about this project, they demonstrated a huge interest in the outputs of this model. In order to please all the community, the data team has to publish their work in a way that everyone with access could see the results. The chosen method was to create a <a href="https://t.me/rmm_rossmann_bot">Telegram Bot</a>.


The data we are going to use is a public dataset that can be found in this <a href="https://www.kaggle.com/competitions/rossmann-store-sales/data">Kaggle Dataset</a>.

|Attribute | Definition
------------ | -------------
|store | unique ID for each store|
|days_of_week | weekday, starting 1 as Monday. |
|date | date that the sales occurred |
|sales | amount of products or services sold in one day  |
|customers | number of customers |
|open | whether the store was open (1) or closed (0)|
|promo | whether the store was participating on a promotion (1) or not (0)|
|sate_holiday | whether it was a state holiday (a=public holiday, b=easter holiday, c=christmas) or not (0) |
|store_type | designates the store model as a, b, c or d. |
|assortment | indicates the store assortment as: a=basic, b=extra, c=extended |
|competition_distance | distance in meters to the nearest competitor store |
|competition_open_since_month | the approximate month competitor was opened |
|competition_open_since_year | the approximate year competitor was opened |
|promo2 | whether the store was participating on a consecutive promotion (1) or not (0)|
|promo2_since_week | indicates the calendar week the store was participating in promo2 |
|promo2_since_year | indicates the year the store was participating in promo2 |
|promo2_interval | indicates the intervals in which promo2 started |



# 2.0 - Business Assumptions

We need to stablish some premises before we move on into the analysis:

- Stores without a competition distance was considered that the competition was very far away. We are going to use this assumption on NA treatment.
- Following the idea above, stores with no competition_since will receive the date when the store has made its first selling.
- The days in which a store was closed, was removed, since it won't add useful information to our model.
- The feature customers was not used in our model, because this feature itself is another prediction problem.

# 3.0 - Solution Strategy

The development of the solution followed the CRISP-DM/DS methodology, with some minor changes (since we are not building a ML model in this project):

1. **Business problem:** Predict the future sales from every Rossmann store, in order to properly invest the company resources;

2. **Business understanding:** How many stores exists, the type of each store, the assortment level, the impacts of competitors nearby, how holidays and promos affects the sales;

3. **Data extraction:** Collect data from Kaggle and import it using a jupyter notebook.

4. **Data cleaning:** Use python and its libraries to check for outliers, NA detection/treatment, feature engineering;

5. **Exploratory data analysis:** Use the cleaned dataset to draw charts with matplotlib and seaborn in order to discover insights, answer questions, evaluate hypothesis, uncover hidden information from the data;

6. **Modeling:** Use the knowledge acquired in EDA to select the features to our model. In this cycle we are also using the Boruta methodology to doublecheck the chosen features. We also prepare these features using normalization, rescaling and transformation.

7. **ML Algorithms:** We will build different regression models based on our dataset. To make sure our solution is robust, not overfitting and generalizable to not seen data, we are going to use the cross-validation method. Models to be built: A baseline with average, linear regression, Lasso regression, random forest regressor and XGBoost regressor.

8. **Evaluation:** Compare the models results regarding the error metrics (MAE, MAPE, RMSE), translate these results in a business language so the other teams can better understand and form their opinion. If the overall result is good enough we proceed to the deployment phase. If the teams decides to change any aspect of the project, another cycle of CRISP-DM/DS will take place.

8. **Deployment:** Use Heroku platform to deploy our model, which will respond to requests via API. We are also going to host our <a href="https://t.me/rmm_rossmann_bot">Telegram Bot</a> on Heroku.

# 4.0 - Top Data Insights

- On holidays, sales are 34.78% higher than regular days.
- On average, stores with higher assortment have higher sales
- There is a positive relation between the time on promo and the sales. While stores on promo tends to sell more, stores not in promo tends to keep its average sales.
- Stores with near competitors tends to sell the same (in some cases, tends to sell more) than stores with far competitors.
- Sales increased by 5% on winter, with the higher peak in December.

# 5.0 - Machine Learning Models

Before we start building our models we set the minimum performance model. We will call it baseline model. It will serve as a guide, to check whether a built model is good or not. At the end of the process we will run a hyperparameter tuning in order to enhance our model.

## 5.1 - Models Performance

**Baseline Model**

| Model Name | MAE    | MAPE      | RMSE |
|-----------|---------|-----------|---------|
|  Average Model  | 1354.80 | 0.20   | 1835.13 |

**Performance with Cross Validation**

| Model Name | MAE    | MAPE      | RMSE |
|-----------|---------|-----------|---------|
|  Linear Regression  | 2105.36 +/- 291.3	| 0.31 +/- 0.02	| 2975.66 +/- 479.35 |
|  Lasso Regression	  | 2126.92 +/- 365.46 |	0.3 +/- 0.01 |	3056.74 +/- 547.17 |
|  Random Forest Regressor | 981.12 +/- 346.29 |	0.14 +/- 0.04 |	1409.5 +/- 499.37 |
|  XGBoost Regressor	  | 972.5 +/- 330.82 |	0.13 +/- 0.04 |	1412.93 +/- 498.84 |

**Final Model (After Hyperparameter Tuning)**

| Model Name | MAE    | MAPE      | RMSE |
|-----------|---------|-----------|---------|
|  XGBoost Regressor  | 689.67	 | 0.098   | 1018.42 |

When we compare the results between the built models, we will see that both linear and lasso regressions performed very poorly, even worse than our baseline model. But, the random forest and the XGBoost showed a very good results. The latter has the best result, so we will choose it.

Once we have selected our model, we will run a hyperparameter tuning to improve even more our model, so we can get the best results to our project. As we can see we successfully enhanced our model.


## 5.2 - Overall Business Results

To achieve a better communication among the teams, it is a good practice to translate all technical vocabulary into a better understanding one. In this case, we translated the metrics from the models to financial values. The business and financial teams appreciate it!

| Scenario | Values    |
|-----------|---------|
|  Predictions  | US$ 285,424,832.00	|
|  Worst Scenario	  |US$ 284,651,793.79   |
|  Best Scenario |US$ 286,197,814.45 |

This is a general expected income from all stores. The individual results will be accessible through the Telegram, as said before.

# 6.0 - Conclusions

After all this steps, the meeting with the CEO and other the other teams will occur. In this meeting will be decided whether to improve the project / model, or finish this project to start a new one. For now, everyone will have access to the model output through the Telegram and all allocated resources will be, for sure, way more assertive than before.

<img src="telegram3.jpg" alt="logo" style="zoom:70% ;" />
<img src="telegram2.jpg" alt="logo" style="zoom:65% ;" />
<img src="telegram1.jpg" alt="logo" style="zoom:70% ;" />

# 7.0 - Next steps

- Improve the functionalities of the telegram bot.
- Try different approaches for NA cleaning.
- Gather information about the number of citizens that lives around each store.
- Ask the business and product teams to elaborate a list of other features that are important for the business.
- Gather information of how many competitors are nearby.
- Use the gathered information to build a better model.  

# 8.0 - Used Tools

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)

![Heroku](https://img.shields.io/badge/heroku-%23430098.svg?style=for-the-badge&logo=heroku&logoColor=white)
