My final implementation in the API (api.py) uses an ensemble of an XGBoost model (xgb_best_model.pkl) and a Neural Network (nn_best_model.pkl) to make a prediction.

In all, the ensemble model achieves an R2 score of 0.8515 on the test set as seen in the colab notebook (CSoT_Dev_+_AI_Week_2.ipynb)

The final features used are derived from the metadata and the content, the hour of the day, the day of the week, and the inferred companies were one hot encoded.

In the API, I ended up using a Gemini call to map the username to one of the 220 'inferred companies'

In addition to this, there are 1000 top tf_idf words are features, along with a TextBlob sentiment (polarity and subjectivity)

This gave me an R2 score of 0.7827 using Linear Regression, which didn't improve much with L2 regularization. Then I tried Random Forest models, barely achieving better performance than Linear
Regression with an R2 score of 0.7884 with 400 n_estimators and a max_depth = 30

The XGBoost model had the best individual R2 score of 0.8484, and all these reported R2 scores are on the 'dev' set while all the models were trained on the 'train' set

Finally, I tried training a Neural Network with the data, finally settling on 6 layers after much tinkering, giving me an R2 score of 0.8311 on the 'dev' set

At last, I tried considering both their predictions just for fun and ended up finding weights for their weighted average that give the best R2 score (0.8520 on the 'dev' set and 0.8515 on the 'test' set 
indicating that the model isn't overfit to the 'dev' set either)
