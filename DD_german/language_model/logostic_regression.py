import h5py
import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
import sys
from lazypredict.Supervised import LazyClassifier
from sklearn import preprocessing
if __name__ == "__main__":
    max_iter = 10000
    if len(sys.argv) == 1:
        print("Usage: python logiscticRegression.py withoutmotion || python logiscticRegression.py withmotion ")
        sys.exit(0)
    elif sys.argv[1] == "withoutmotion":
        print(f"*************************** without motion data AND max_iter = {max_iter} ***********************")
        disflueny_data = h5py.File('./word_tag_csv_word_dur2/dataset.h5py', 'r') # without motion data
        
    elif sys.argv[1] == "withmotion":
        print(f"*************************** with motion data AND max_iter = {max_iter} ************************")
        disflueny_data = h5py.File('./disf_tags_word_level_w_motion/dataset.h5py', 'r') # with motion data
        

    X = disflueny_data['feature_vetor'][...]
    y = disflueny_data['target_vector'][...]
    Y = list()
    for index, encode in enumerate(y):
        Y.append(np.argmax(encode))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    

    clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=classification_report)
    models,predictions = clf.fit(X_train, X_test, y_train, y_test)
    # print(models)
    # print()
    print(predictions)
    # logreg = LogisticRegression(random_state=42,max_iter=max_iter)
    # logreg.fit(X_train, y_train)
    
    # y_pred = logreg.predict(X_test)
    # classification_matrix = classification_report(y_test, y_pred)
    # print(classification_matrix)
    
