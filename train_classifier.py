import sys
import pandas as pd
import re
import pickle
import nltk
nltk.download(['punkt', 'stopwords', 'wordnet'])
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    '''Load data from sqldatabase
    
    Returns:
    X: feature variables
    y: target variables
    categories_names (list): column's name
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM messages', con=engine)
    X = df['message']
    y = df.iloc[:,4:]
    y['related'].replace(2, 1, inplace=True)
    category_names = y.columns.tolist()
    return X, y, category_names

def tokenize(text):
    '''Reduce sentences to its original words'''
    text = re.sub(r'[^\w\s]','',text.lower())
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()
    clean_tok = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stopwords.words("english")]
    return clean_tok


def build_model():
    '''Build a machine learning pipeline to train and search for best parameters'''
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)), 
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultiOutputClassifier(AdaBoostClassifier()))])
    
    parameters = {'vect__max_df': [0.75, 1.0]}
    
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3, cv=2, n_jobs=-1)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''Get the precision, recall, f1 score of model'''
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns=category_names)
    evaluation = {}
    for column in Y_test.columns:
        evaluation[column] = []
        evaluation[column].append(precision_score(Y_test[column], y_pred_df[column], average = 'micro'))
        evaluation[column].append(recall_score(Y_test[column], y_pred_df[column], average = 'micro'))
        evaluation[column].append(f1_score(Y_test[column], y_pred_df[column], average = 'micro'))
    print(pd.DataFrame(evaluation))


def save_model(model, model_filepath):
    '''save model to pickle file'''
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
