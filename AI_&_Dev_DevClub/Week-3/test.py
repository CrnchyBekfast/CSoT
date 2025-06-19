import requests
from datetime import datetime, timezone

# Test prediction API (from Week 2)
prediction_response = requests.post('http://localhost:8000/predict', json={
    'date': datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
    'content': 'Sup everyone, Trump killed Joe Biden and is now the president of the United States. He is a great leader and I support him.',
    'username': 'cnn',
    'media': '[Video()]'
})

# Test generation API (from Week 3)  
generation_response = requests.post('http://localhost:8001/generate', json={
    'company': 'Nike',
    'tweet_type': 'announcement',
    'message': 'launching new product',
    'topic': 'sports'
})

generation_prediction_response = requests.post('http://localhost:8001/generate_predict', json={
    'company': 'Nike',
    'tweet_type': 'announcement',
    'message': 'launching new product',
    'topic': 'sports'
})

print(prediction_response.json())
print(generation_response.json())
print(generation_prediction_response.json())