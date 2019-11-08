import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

import warnings
warnings.filterwarnings("ignore")

class ModelAll:
    
    def __init__(self, X_train, X_test, y_train, y_test, 
                    LogisticRegression, RandomForestClassifier, GradientBoostingClassifier):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.LogisticRegression = LogisticRegression()
        self.RandomForestClassifier = RandomForestClassifier()
        self.GradientBoostingClassifier = GradientBoostingClassifier()

    def whole_shebang(self, model):
        clf = model
        clf.fit(self.X_train, self.y_train)
        return clf.score(self.X_test, self.y_test)

    def grid_it(self):
        pass

    def run_it(self):
        model_list = [self.LogisticRegression, self.RandomForestClassifier, self.GradientBoostingClassifier]
        for i in model_list:
            print (self.whole_shebang(i))

def get_and_clean_data():
    df = pd.read_csv('data/list_attr_celeba.csv')
    df.columns = map(str.lower, df.columns)
    df.replace([-1], 0, inplace=True)
    return df

if __name__ == "__main__":
    df_labels = get_and_clean_data()
    print ('Retrieved and cleaned_data!')
    X_train, X_test, y_train, y_test = train_test_split(df.drop(['male', 'image_id'], axis=1), df.male)
    test = ModelAll(X_train, X_test, y_train, y_test, 
                    LogisticRegression, RandomForestClassifier, GradientBoostingClassifier)
    test.run_it()