import pandas as pd
from sklearn import svm
from sklearn import decomposition
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier 

def formatData2():
    train_data = pd.read_csv('data.csv')
    test_data = pd.read_csv('quiz.csv')
    train_data_without_type = train_data[train_data.columns[:52]]
    frames = [train_data_without_type, test_data]
    all_data = pd.concat(frames)
    print "train + test data shape: " + str(all_data.shape)
    category_cols = []
    num_cols = []
    for colName in all_data.columns:
        col = all_data[colName]
        if col.dtype == 'object':
            category_cols.append(col)
        else:
            num_cols.append(col)
    category_df = pd.concat(category_cols, axis = 1, keys=[s.name for s in category_cols])
    print 'category_cols number is: ' + str(len(category_df.columns))
    num_df = pd.concat(num_cols, axis = 1, keys=[s.name for s in num_cols])
    print 'category_cols number is: ' + str(len(num_df.columns))
    dummy_data = pd.concat([pd.get_dummies(category_df[col]) for col in category_df], axis=1)
    print 'dummy_data cols number is: ' + str(len(dummy_data.columns))

    formatted_data = dummy_data
    for colName in num_df.columns:
        formatted_data[colName] = num_df[colName]

    format_train_data = formatted_data[:train_data.shape[0]]
    format_train_data['label'] = train_data['label']
    format_test_data = formatted_data[train_data.shape[0]:]

    return format_train_data, format_test_data


def train_random_forests(train_data):
    forest = RandomForestClassifier(n_estimators = 100)
    feature = train_data.loc[:, :'64']
    label = train_data['label']
    print "the number of features is: " + str(len(feature.columns))
    forest = forest.fit(feature, label)
    return forest

def test_random_forests(forest, test_data):
    return forest.predict(test_data)
    