from flask import Flask, render_template, url_for, request

app = Flask(__name__)

# Main page
@app.route('/', methods=['GET','POST'])
def index():
    title = "Hotel Review Sentiment Analysis"
    if request.method == 'GET':
        return render_template('cover.html', title=title)
    else:
        import string
        from string import punctuation
        from nltk.corpus import stopwords
        import pickle
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.feature_extraction.text import TfidfTransformer
        from tensorflow.keras.models import model_from_json, load_model
        import numpy as np
        def get_text_processing(text):
            stpword = stopwords.words('english')
            no_punctuation = [char for char in text if char not in string.punctuation]
            no_punctuation = ''.join(no_punctuation)
            return ' '.join([word for word in no_punctuation.split() if word.lower() not in stpword])

        def get_class(preds):
            index = np.argmax(preds, axis=1)
            if index == 0:
                label = 'Negative'
            elif index == 1:
                label = 'Neutral'
            elif index == 2:
                label = 'Positive'
            return label
        
        text_review = request.form['review']
        review = request.form['review']
        review = [get_text_processing(review)]
        review = np.array(review)

        feature_path = 'model/feature_24.pkl'
        vect =  CountVectorizer(vocabulary=pickle.load(open(feature_path, "rb")))

        tfidf_path = 'model/tfidftransformer_24.pkl'
        tfidf = pickle.load(open(tfidf_path, "rb"))

        review = vect.transform(review)
        review = tfidf.transform(review)
        review = review.toarray()

        loaded_model = load_model("model/tripadvisor_hotel_reviews_full_24_min.h5")
        # opt=tf.keras.optimizers.Adam(learning_rate=0.001)
        # loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        # # load json and create model
        # json_file = open('model/model_3000_30_min.json', 'r')
        # loaded_model_json = json_file.read()
        # json_file.close()
        # loaded_model = model_from_json(loaded_model_json)
        # # load weights into new model
        # loaded_model.load_weights("model/tripadvisor_hotel_reviews_3000_30_min.h5")

        preds = loaded_model.predict(review)
        sentiment = get_class(preds)
        
        return render_template('cover.html', title=title, review=text_review, sentiment=sentiment)

if __name__ == '__main__':
    app.run()