from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tweet_generator import SimpleTweetGenerator

app = FastAPI()

class Metadata(BaseModel):
    company: str
    tweet_type: str = "general"
    message: str = "Something awesome!"
    topic: str = "innovation"

@app.post("/generate")
async def generate(metadata: Metadata):
    generator = SimpleTweetGenerator()
    
    try:
        tweet = generator.generate_tweet(
            company=metadata.company,
            tweet_type=metadata.tweet_type,
            message=metadata.message,
            topic=metadata.topic
        )
        return {"tweet": tweet}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/generate_ai")
async def generate_ai(metadata: Metadata):
    from ai_tweet_generator import generate
    try:
        tweet = generate(
            company=metadata.company,
            tweet_type=metadata.tweet_type,
            message=metadata.message,
            topic=metadata.topic
        )
        return {"tweet": tweet}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/generate_predict")
async def generate_predict(company: str, tweet_type: str = "general", message: str = "Something awesome!", topic: str = "innovation"):
    from ai_tweet_generator import generate
    from api import predict
    from datetime import datetime
    import requests
    try:
        tweet = generate(
            company=company,
            tweet_type=tweet_type,
            message=message,
            topic=topic
        )
        
        # Simulate a prediction call
        prediction_response = requests.post('http://localhost:8000/predict', json={
        'date': '2025-06-19T09:37:37.731Z',
        'content': tweet,
        'username': company,
        'media': '[Video()]'
        })
        
        return {"tweet": tweet, "prediction": prediction_response.json().get("predicted_value")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


