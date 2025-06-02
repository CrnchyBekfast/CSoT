The dataset has been cleaned, along with feature engineering making it ready to use for training any model on the data.

Here is the google drive link to the cleaned dataset

[Cleaned Dataset](https://drive.google.com/file/d/1-6SZkZfpSa9k-UJqQo47wFXGuB-qwlFO/view?usp=sharing)

The set of features I have arrived at are the example ones, i.e. word_count and char_count. Apart from this, the datetime features that were extracted from the
date data, i.e. day of the week, and hour have also been one-hot encoded

The inferred companies data has also been one hot encoded, giving 220 additional features

Through textBlob, we have 2 new features for the content, i.e. the sentiment and polarity which may help in predicting the popularity of a tweet

In addition, TF-IDF vectorization has also been implemented, for now for the top 1000 words in the dataset 'content' field, giving 1000 additional features.

Looking at the distribution of the likes, which was very sparse at the higher end, the log_likes field has been created, which will act
as the target for the model, and it's prediction can just be used in e^(predicted_value) to get the predicted no. of likes.

In the proposed API, we can take the metadata and content as inputs, i.e the datetime, the text, the username and the link to the media attachment

Then using the saved one-hot encoders, and tfidf_vectorizer, we can bring the data into a usable form, run it through the model, and output
the prediction to the end user.

One thing that will need to be figured out is how to confidently map the username to one of the 220 fields in the inferred_companies column.
Perhaps regex may suffice.
