An api for tweet generation using templates, and the gemini api has been created, and is in main.py

The file test.py contains the code for simultaneous testing of both the week-2 api (api.py) with all its dependencies that have not been included here because they 
would be redundant, and the week-3 api (main.py) that have to both be run using uvicorn on different ports

Following the docs somewhat, I used the ports 8000 and 8001, by simply adding --port 800x to uvicorn api:app --reload, or uvicorn main:app --reload in the terminal

As mentioned above, apart from the template based tweet generator, a more personalized and better tweet generation logic has been implemented using the same input 
features using the gemini-2.0-flash model instead of GPT-2 as in the documentation.
