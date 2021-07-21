from MBC_exploration import *

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

if "__main__" == __name__:
    trainset_path = sys.argv[1]
    evalset_path = sys.argv[2]
    output_path = sys.argv[3]

    training = MBC_Training(trainset_path, evalset_path)
    best_SVM()

    write_to_file(output, output_path)