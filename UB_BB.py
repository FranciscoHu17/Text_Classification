import sys
import os
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

class UB_BB_Training:
    """ Used to train models based on the given training path and evaluation path
        
        Atrributes:
        categories  Categories of the training and evaluation data
        data_train  Files loaded from train_path that will be used to train the models
        data_test   Files loaded from eval_path that will be used to test the models
        LC          Used to store the Learning Curve of each model
        count_vect  CountVectorizer used to transform the data
        X_train     Training data
        Y_train     Training targets
        X_test      Testing data
        Y_test      Testing targets   
        predicted   Model's predictions 
    """
    def __init__(self, train_path, eval_path):
        categories = os.listdir(train_path)
        self.data_train = load_files(train_path, categories=categories, encoding="ISO-8859-1")
        self.data_test = load_files(eval_path, categories=categories, encoding="ISO-8859-1")
        self.LC = {}
    
    # Splits the training data based on the size
    def split_data(self, train_size: int=None):
        if train_size is not None:
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.data_train.data, self.data_train.target , train_size=train_size)
        else:
            self.X_train = self.data_train.data
            self.Y_train = self.data_train.target
        self.X_test = self.data_test.data 
        self.Y_test = self.data_test.target 

    # Transforms the training and test data based on ngram using features from CountVectorizer
    def transform(self, ngram: int):
        self.count_vect = CountVectorizer(ngram_range=(ngram,ngram)) 
        self.X_train = self.count_vect.fit_transform(self.X_train)
        self.X_test = self.count_vect.transform(self.X_test)

    # Adds the correct classifier and train the model by fitting it
    def train_model(self, classifier: str):
        if classifier == "NB": self.pipe = make_pipeline(ComplementNB())
        elif classifier == "LR": self.pipe = make_pipeline(LogisticRegression(max_iter=500)) 
        elif classifier == "SVM": self.pipe = make_pipeline(StandardScaler(with_mean=False), LinearSVC(dual=False)) 
        elif classifier == "RF": self.pipe = make_pipeline(RandomForestClassifier())

        self.pipe.fit(self.X_train, self.Y_train)
        
    # Transforms, trains, then predicts the model based on the data given
    def predict(self, classifier: str, ngram: int):
        self.transform(ngram)
        self.train_model(classifier)
        self.predicted = self.pipe.predict(self.X_test)
    
    # Evaluates the performance of the prediction
    def get_evaluation(self):
        return metrics.precision_recall_fscore_support(self.Y_test, self.predicted, average='macro')
    
    # Gets the f1 score of the prediction
    def get_f1(self):
        return metrics.f1_score(self.Y_test, self.predicted, average='macro')

    # Samples the data to produce a Learning Curve
    def sample_LC(self, classifier: str, ngram: int, all_data_f1: int):
        sizes = [.1,.2,.3,.4,.5,.6,.7,.8,.9]
        f1 = []
        for sample in sizes:
            self.split_data(sample)
            self.predict(classifier, ngram)
            f1.append(self.get_f1())
        f1.append(all_data_f1)
        sizes.append(1)
        self.LC[classifier] = (sizes, f1)

    # Plot the Learning Curve
    def plot(self):
        axes = plt.gca()
        axes.set_title("Learning Curve (LC)")
        axes.set_xlabel("Size of Training Data (%)")
        axes.set_ylabel("F1-Score")
        sample_sizes = None

        for classifier in self.LC:
            sizes = [100*sample for sample in self.LC[classifier][0]]
            f1_scores = self.LC[classifier][1]
            plt.plot(sizes,f1_scores,'o-',label=classifier)
            sample_sizes = sizes

        plt.legend(loc="lower right")
        plt.xticks(sample_sizes)
        plt.show()

        

    def print_accuracy(self):
        #print("Accuracy: ", metrics.accuracy_score(self.Y_test, self.predicted))
        print(metrics.classification_report(self.Y_test, self.predicted, target_names=self.data_test.target_names))
        #print(metrics.confusion_matrix(self.Y_test, self.predicted))

def write_to_file(output, path):
    file = open(path,"a")
    file.truncate(0)
    file.write(output)
    file.close() 

def run_baselines():
    output = ""
    for classifier in baselines:
        for ngram, config in enumerate(["UB", "BB"],start=1):
            train.split_data()
            train.predict(classifier, ngram)
            evals = train.get_evaluation()
            run = classifier+","+config+","+str(evals[0])+","+str(evals[1])+","+str(evals[2])
            output+=run+"\n"
            print("Ran "+classifier+" classifier with ngram of "+str(ngram)+" and its output is: "+run)

            if display_LC == 1 and ngram == 1:
                print("Recording the Learning Curve for "+classifier+" with ngram of "+str(ngram))
                train.sample_LC(classifier, ngram, evals[2])
        print()      
    write_to_file(output, output_path)

if "__main__" == __name__:
    trainset_path = sys.argv[1]
    evalset_path = sys.argv[2]
    output_path = sys.argv[3]
    display_LC = int(sys.argv[4])

    train = UB_BB_Training(trainset_path, evalset_path)
    baselines = ["NB","LR","SVM","RF"] 
    run_baselines()

    if display_LC == 1:
        print("Plotting the graph of the Learning Curve")
        train.plot()