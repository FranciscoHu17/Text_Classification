import sys
import os
from nltk.stem.porter import *
from sklearn.datasets import load_files
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn import metrics

output = ""

class MBC_Training:
    def __init__(self,train_path, eval_path):
        categories = os.listdir(train_path)
        self.data_train = load_files(train_path, categories=categories, encoding="ISO-8859-1")
        self.data_test = load_files(eval_path, categories=categories, encoding="ISO-8859-1")
        self.X_train = self.data_train.data
        self.Y_train = self.data_train.target
        self.X_test = self.data_test.data 
        self.Y_test = self.data_test.target
        self.vectorizer =  None 
        self.predicted = None
        self.model = None
        self.steps = []   
        


    def stem_words(self):
        stemmer = PorterStemmer()
        self.X_train = [" ".join([stemmer.stem(w) for w in doc.split(" ")]) for doc in self.X_train]
        self.X_test = [" ".join([stemmer.stem(w) for w in doc.split(" ")]) for doc in self.X_test]


    def vectorizer_transform(self, vectorizer, ngram_range=(1,1), stop_words=None):  
        if vectorizer == "count":
            self.vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words)
        elif vectorizer == "tfid":
            self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words=stop_words)
        
        self.X_train = self.vectorizer.fit_transform(self.X_train)
        self.X_test = self.vectorizer.transform(self.X_test)

    
    

    def add_pipeline_step(self, step):
        if step == "NB": self.steps.extend([(step,ComplementNB())])
        elif step == "LR": self.steps.extend([(step,LogisticRegression(C=.5,max_iter=500))]) 
        elif step == "SVM": self.steps.extend([("Select", SelectPercentile(chi2,percentile=5.5)),("Scaler",StandardScaler(with_mean=False, with_std=False)),
                                               (step,LinearSVC(C=4.5,max_iter=10000,class_weight='balanced',dual=False, intercept_scaling=1.1))]) 
        elif step == "RF": self.steps.extend([("Normal",Normalizer(norm='l1')),("Scaler",StandardScaler(with_mean=False)),
                                            (step,RandomForestClassifier(n_estimators = 500, n_jobs = -1,class_weight="balanced",min_samples_split=.1,bootstrap=False,min_samples_leaf=5))])

    def train_model(self):
        self.model = Pipeline(self.steps)
        self.model.fit(self.X_train, self.Y_train)
    
    def predict_results(self):
        self.predicted = self.model.predict(self.X_test)
    
    def reset_training(self):
        self.X_train = self.data_train.data
        self.Y_train = self.data_train.target
        self.X_test = self.data_test.data 
        self.Y_test = self.data_test.target
        self.steps = []
    
    def get_evaluation(self):
        return metrics.precision_recall_fscore_support(self.Y_test, self.predicted, average='macro')

    
def best_NB():
    #training.stem_words()
    training.vectorizer_transform("count", ngram_range=(1,2), stop_words='english')   
    training.add_pipeline_step("NB") 
    training.train_model()
    training.predict_results()
    evals = training.get_evaluation()
    training.reset_training()
    run = "NB,UBBB,"+str(evals[0])+","+str(evals[1])+","+str(evals[2])
    global output
    output += run + "\n"

    print(evals)

def best_LR():
    training.vectorizer_transform("count", ngram_range=(1,1),stop_words='english')   
    training.add_pipeline_step("LR") 
    training.train_model()
    training.predict_results()
    evals = training.get_evaluation()
    training.reset_training()
    run = "LR,UB,"+str(evals[0])+","+str(evals[1])+","+str(evals[2])
    global output
    output += run + "\n"
    print(evals)

def best_SVM():
    training.stem_words()
    training.vectorizer_transform("tfid", ngram_range=(1,2))   
    training.add_pipeline_step("SVM") 
    training.train_model()
    training.predict_results()
    evals = training.get_evaluation()
    training.reset_training()
    print(evals)
    run = "SVM,UBBB,"+str(evals[0])+","+str(evals[1])+","+str(evals[2])
    global output
    output += run + "\n"

def best_RF():
    #training.stem_words()
    training.vectorizer_transform("tfid", ngram_range=(1,1), stop_words='english')   
    training.add_pipeline_step("RF") 
    training.train_model()
    training.predict_results()
    evals = training.get_evaluation()
    training.reset_training()
    run = "RF,UB,"+str(evals[0])+","+str(evals[1])+","+str(evals[2])
    global output
    output += run + "\n"
    print(evals)

def write_to_file(output, path):
    file = open(path,"a")
    file.truncate(0)
    file.write(output)
    file.close() 

if "__main__" == __name__:
    trainset_path = sys.argv[1]
    evalset_path = sys.argv[2]
    output_path = sys.argv[3]

    training = MBC_Training(trainset_path, evalset_path)
    best_NB()
    best_LR()
    best_SVM()
    best_RF()

    write_to_file(output, output_path)
    