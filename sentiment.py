import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
#nltk.download("stopwords")          # Download the stop words from nltk
import itertools
import collections


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    #print train_pos
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg



def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))
    #print stopwords

    # Determine a list of words that will be used as features. 
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    
       #get all the words without duplicates
    train_words = set(list(itertools.chain(*train_neg))+list(itertools.chain(*train_pos)))
    
        #eliminated the duplicate inside each text
    pos = []
    neg = []
    for text in train_pos:
        temp = set(text)
        pos.append(list(temp))
    for text in train_neg:
        temp = set(text)
        neg.append(list(temp))

         #map the word into count
    temp_pos = list(itertools.chain(*pos))
    temp_neg= list(itertools.chain(*neg))
    pos_dic = collections.Counter(temp_pos)
    neg_dic = collections.Counter(temp_neg)

    dic1 = {x:y for x,y in pos_dic.iteritems() if y>=int(len(train_pos)*0.01) and x not in stopwords}
    dic2 = {x:y for x,y in neg_dic.iteritems() if y>=int(len(train_neg)*0.01) and x not in stopwords}

    condition12 = dic1.viewkeys() | dic2.viewkeys()
    intersection = pos_dic.viewkeys() & neg_dic.viewkeys()

    condition3 = []
    for key in intersection:
        if pos_dic[key]>=2*neg_dic[key] or neg_dic[key]>=2*pos_dic[key]:
            condition3.append(key)
    features = set(condition12) & set(condition3)

    #print features
    print len(features)

    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    # YOUR CODE HERE
    train_pos_vec = vec_map(features,train_pos)
    train_neg_vec = vec_map(features,train_neg)
    test_pos_vec = vec_map(features,test_pos)
    test_neg_vec = vec_map(features,test_neg)
    print len(test_neg)
    print len(test_neg_vec)




    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec

def vec_map(features,textset):
    vectors = []
    for text in textset:
        temp = []
        for feature in features:
            if feature in text:
                temp.append(1)
            else:
                temp.append(0)
        vectors.append(temp)
    return vectors


def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    # YOUR CODE HERE
    labeled_train_pos = obj_map(train_pos,'TRAIN_POS_')
    labeled_train_neg = obj_map(train_neg,'TRAIN_NEG_')
    labeled_test_pos = obj_map(test_pos,'TEST_POS_')
    labeled_test_neg = obj_map(test_neg,'TEST_NEG_')


    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run 
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)
    # Use the docvecs function to extract the feature vectors for the training and test data
    # YOUR CODE HERE
    train_pos_vec = vec_map_doc(train_pos,model,'TRAIN_POS_')
    train_neg_vec = vec_map_doc(train_neg,model,'TRAIN_NEG_')
    test_pos_vec = vec_map_doc(test_pos,model,'TEST_POS_')
    test_neg_vec = vec_map_doc(test_neg,model,'TEST_NEG_')




    
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec

def obj_map(list,listname):
    labeled = []
    for i, text in enumerate(list):
        obj =  LabeledSentence(words=text, tags=[listname+str(i)])
        labeled.append(obj)
    return labeled

def vec_map_doc(list,model,listname):
    feature_vec = []
    for i in range(0,len(list)):
        vector = model.docvecs[listname+str(i)]
        feature_vec.append(vector)
    return feature_vec




def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    X = train_pos_vec+train_neg_vec
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)
    clf1 = sklearn.naive_bayes.BernoulliNB()
    nb_model = clf1.fit(X,Y)
    clf2 = sklearn.linear_model.LogisticRegression()
    lr_model = clf2.fit(X,Y)

    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    
    return nb_model, lr_model


def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)
    
    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    
    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    # YOUR CODE HERE
    pos_temp = model.predict(test_pos_vec)
    neg_temp = model.predict(test_neg_vec)
    pos_predict = collections.Counter(pos_temp)
    neg_predict = collections.Counter(neg_temp)
    tp = pos_predict['pos']
    tn = neg_predict['neg']
    fn = pos_predict['neg']
    fp = neg_predict['pos']
    accuracy = (tp+tn)/float(tp+tn+fn+fp)
    #print pos_predict
    #print neg_predict
    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()
