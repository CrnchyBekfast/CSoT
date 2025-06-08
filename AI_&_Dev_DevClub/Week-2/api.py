from typing import Union
from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import re
import google.generativeai as genai
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler

app = FastAPI()

xgb_model = joblib.load('xgboost_best_model.pkl')
nn_model = joblib.load('nn_best_model.pkl')
day_encoder = joblib.load('day_encoder.pkl')
hour_encoder = joblib.load('hour_encoder.pkl')
media_type_encoder = joblib.load('media_type_encoder.pkl')
company_encoder = joblib.load('company_encoder.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
scaler = joblib.load('nn_scaler.pkl')

GOOGLE_API_KEY = 'AIzaSyBzAqO3MBc4R0F-o6lDT-tfdzVjxZdFJgI'
genai.configure(api_key =GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-2.0-flash')

def get_inferred_company(company):
    inferred_companies_list = "{'aaa', 'lyft', 'gsa', 'ge healthcare', 'qatar airways', 'lacoste', 'mcafee', 'amazon', 'pm', 'fc barcelona', 'wells fargo', 'sabc', 'taco bell', 'mtn', 'home depot', 'motorola', 'holland america line', 'the home depot', 'gucci', 'sap', 'greenpeace', 'petsmart', 'little caesars pizza', 'ihop', 'sara lee', 'zumba', 'wolseley', 'harris teeter', 'ericsson', 'united airlines', 'charles schwab', 'urban outfitters', 'american family insurance', 'total', 'cbc', 'aarp', 'aviva', 'eaton', 'opel', 'waitrose', 'accenture', 'office depot', 'henry schein', 'toyota', 'aa', 'facebook', 'asus', 'tommy hilfiger', 'rite aid', 'fedex', 'alcoa', 'avon', 'dollar general', 'sutter health', 'target', 'metlife', 'icici bank', 'red lobster', 'amd', 'pepsico', 'bny mellon', 'dsw', 'qantas', 'airtel', 'general mills', 'desjardins', 'arc', 'goldman sachs', 'general motors', 'bayer', 'nokia', 'bar', 'instagram', 'illumina', 'hobby lobby', 'belk', 'rockwell automation', 'spirit airlines', 'pfizer', 'microsoft', 'paytm', 'cisco', 'united rentals', 'lufthansa', 'boeing', 'bic', 'hp', 'metro', 'renault', 'honeywell', 'starbucks', 'herbalife', 'astro', 'cnn', 'depaul university', 'novartis', 'singapore airlines', 'cricket wireless', 'oracle', 'fred hutch', 'hudson', 'singtel', 'food lion', 'arm', 'bell', 'turkish airlines', 'rona', 'staples', 'new balance', 'pnc', 'rotary international', 'getty images', 'axa', 'airasia', 'free', 'movistar', 'meijer', 'ups', 'lenovo', 'continental', 'malaysia airlines', 'buffalo wild wings', 'dupont', 'allstate', 'the body shop', 'trane', 'hubspot', 'shell', 'ibm', 'tim hortons', 'samsung', 'shoprite', 'twitter', 'american express', 'acer', 'chanel', 'tesla', 'comcast', 'no', 'saab', 'national geographic', 'olive garden', 'raytheon', 'talbots', 'paramount pictures', 'puma', 'tesco', 'emerson', 'midea', 'scotiabank', 'hertz', 'sephora', 'coty', 'shutterstock', 'converse', 'lane bryant', 'western union', 'maximus', 'williams', 'hsbc', 'asda', 'heineken', 'state farm', 'westfield', 'usaa', 'groupon', 'independent', 'red bull', 'virgin media', 'alaska airlines', 'unilever', 'yahoo', 'cvs pharmacy', 'equinox', 'restaurant', 'oakley', 'walmart', 'pirelli', 'hitachi', 'revlon', 'travelers', 'hugo boss', 'ymca', 'stryker', 'spotify', 'greene king', 'nn', 'padi', 'mayo clinic', 'saks fifth avenue', 'siemens', 'monster', 'bacardi', 'amp', 'john deere', 'ethiopian airlines', 'amway', 'cameron', 'michelin', 'pizza hut', 'mastercard', 'mcdonalds', 'peugeot', 'td', 'credit suisse', 'amtrak', 'lg electronics', 'louis vuitton', 'ulta beauty', 'philip morris international', 'dairy queen', 'verizon', 'iberia', 'wwf', 'blackberry', 'amc', 'dominion', 'comcast business', 'valspar', 'ucla health'}"
    inferred_company = model.generate_content("Reply with one word, the mapping (which means the element from the 'list') from this given username: (START OF USERNAME)"+ company +"(END OF USERNAME) to this list of inferred companies: (START OF LIST)"+ inferred_companies_list+ "(END OF LIST). If username matches no company, mapping is to 'independent'(keep in mind the independent is lowercase)")
    print(inferred_company.text)
    return inferred_company.text.strip()

def get_sentiment(text):
  analysis = TextBlob(text)
  return analysis.sentiment.polarity, analysis.sentiment.subjectivity

class Tweet(BaseModel):
    date: datetime
    content: str
    username: str
    media: str

@app.post("/predict")
async def predict(tweet: Tweet):
    df = pd.DataFrame([tweet.model_dump()])
    df['content'] = df['content'].astype(str).str.strip().str.lower()
    df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['word_count'] = df['content'].apply(lambda x: len(x.split()))
    df['char_count'] = df['content'].apply(len)

    day_one_hot_encoded = day_encoder.transform(df[['day_of_week']])
    df[day_encoder.categories_[0]] = day_one_hot_encoded

    df['media_type'] = df['media'].apply(
    lambda x: re.search(r'\[(.*?)\(', x).group(1).strip() if pd.notna(x) and re.search(r'\[(.*?)\(', x) else None
    )
    pd.set_option('display.max_columns', None)
    media_type_one_hot_encoded = media_type_encoder.transform(df[['media_type']])
    df[media_type_encoder.categories_[0]] = media_type_one_hot_encoded

    df['inferred company'] = df['username'].apply(get_inferred_company)
    company_one_hot_encoded = company_encoder.transform(df[['inferred company']])
    df[company_encoder.categories_[0]] = company_one_hot_encoded

    df[['polarity', 'subjectivity']] = df['content'].apply(lambda text: pd.Series(get_sentiment(text)))

    tfidf_matrix = tfidf_vectorizer.transform(df['content'])
    prefixed_cols = ['tfidf_' + feat for feat in tfidf_vectorizer.get_feature_names_out()]
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=prefixed_cols)
    df = pd.concat([df, df_tfidf], axis=1)

    hour_one_hot_encoded = hour_encoder.transform(df[['hour']])


    column_names = [f"{str(cat)}" for cat in hour_encoder.categories_[0]]

    hour_ohe_df = pd.DataFrame(
        hour_one_hot_encoded,
        columns=column_names,
        index=df.index  
    )

    df = pd.concat([df, hour_ohe_df], axis=1)

    mean_word_count = 22.46747
    std_word_count = 11.813076665078382
    df['word_count'] = (df['word_count'] - mean_word_count) / std_word_count
    mean_char_count = 147.52472333333333
    std_char_count = 71.5175829986742
    df['char_count'] = (df['char_count'] - mean_char_count) / std_char_count

    prediction_df = df.drop(columns = [ 'date', 'content', 'username', 'media', 'inferred company', 'datetime', 'hour', 'day_of_week', 'media_type'])

    #print(prediction_df.head(1))

    prediction_scaled_df = scaler.transform(prediction_df)

    print(prediction_scaled_df.shape)

    nn_log_prediction = nn_model.predict(prediction_scaled_df)
    xgb_log_prediction = xgb_model.predict(prediction_df)

    log_pred = 0.3*nn_log_prediction.flatten() + 0.7*xgb_log_prediction.flatten()

    predicted_likes = np.expm1(log_pred)

    return {"predicted_value": round(float(predicted_likes[0]))}






