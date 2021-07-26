import gensim
from gensim.utils import simple_preprocess
import spacy
from sklearn import preprocessing
import re

def simple_process(text):
    process_text = gensim.utils.simple_preprocess(text, deacc=True)
    text = ' '.join([word for word in process_text])
    return text

def clean_text(text):
    """
    Removes unuseful data from the text like URLs, punctuations, symbols etc.

    Parameters:
    ----------
    text : the sentence to be cleaned

    Returns:
    -------
    cleaned text
    """
    
    # remove Unicode characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # taking sequences of characters with alphanumeric characters separated by other characters
    text = re.sub(r"[-?!&]",' ',text)
    text = re.sub(r'''["#$%()*+,./:;<=>@[\]^_`{|}~]''','',text)
    
    # remove all numeric charracters
    text = re.sub(r'[0-9]','',text)

    # Remove new line characters
    text = re.sub(r'\s+', ' ', text)

    # remove all numeric charracters
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '',text)

    Apos_dict={"'s":" is","n't":" not","'m":" am","'ll":" will",
           "'d":" would","'ve":" have","'re":" are"}
  
    #replace the contractions
    for key,value in Apos_dict.items():
        if key in text:
            text=text.replace(key,value)

    # convert to lowercase to maintain consistency
    text = text.lower()
       
    return text

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation

    Lemmatize the text using the allowed POS tags

    Parameters:
    ----------
    text            : the sentence to be cleaned
    allowed_postags : the POS tags that are allowed in the text.

    Returns:
    -------
    lemmatized text
    """
    
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    # Parse the sentence using the loaded 'en' model object `nlp`
    doc = nlp(texts)

    text = ' '.join([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return text

def normalize_and_scale(X_train, X_test):

    # normalize the training and testing data
    X_train_bert = preprocessing.normalize(X_train)
    X_test_bert = preprocessing.normalize(X_test)

    # standardize the testing and training data
    scaler_Xtrain_bert = preprocessing.StandardScaler().fit(X_train_bert)
    scaler_Xtest_bert = preprocessing.StandardScaler().fit(X_test_bert)

    X_train_bert_scaled = scaler_Xtrain_bert.transform(X_train_bert)
    X_test_bert_scaled = scaler_Xtest_bert.transform(X_test_bert)

    return X_train_bert_scaled, X_test_bert_scaled
