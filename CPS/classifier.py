from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def DecisionTree(train_pattern,train_label,test_pattern):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_pattern, train_label)
    result = clf.predict(test_pattern)
    return result


def SVM(train_pattern,train_label,test_pattern):
    clf = SVC(kernel = 'rbf',probability = True, gamma = 'scale')
    clf.fit(train_pattern,train_label)
    result = clf.predict(test_pattern)
    return result


def RF(train_pattern, train_label, test_pattern):
    clf =  RandomForestClassifier(n_estimators=8)
    clf.fit(train_pattern, train_label)
    result = clf.predict(test_pattern)
    return result
