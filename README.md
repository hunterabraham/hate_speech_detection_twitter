# hate_speech_detection_twitter
## Summary
main.py is a script to detect hate speech on twitter using logistic regression with a dataset of labeled tweets to predict hate speech.

## Natural Language Processing
Cleans phrases of twitter API noise (E.g. @, RT, twitter handles), then tokenizes phrases and removes stop words. Then converts words into lemmas.

After the phrases have been cleaned, they are analyzed by frequency using TF-IDF.

## Logistic Regression

Finally, the TF-IDF analysis is fed into a logistic regression model using 70% of the data for training and 30% of the data for testing.

## Results

The resultant accuracy is consistently ~94%.
