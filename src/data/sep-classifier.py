# import libraries
from pyod.utils.data import generate_data
from pyod.models.ocsvm import OCSVM
from pyod.utils.example import visualize  # generate dataset

def accuracy(a, b):
    n = range(len(a))
    return sum([a[i] != b[i] for i in n])

X_train, X_test, y_train, y_test = generate_data(
    n_train=500, n_test=100, n_features=3, behaviour="new"
)  # initialize detector

# X_train = X_train[:5000]
# y_train = y_train[:5000]
# print(y_train)

clf = OCSVM(nu=0.9, gamma=0.5, kernel="rbf", verbose=1)
clf.fit(X_train)  # binary labels
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict_confidence(X_test)  # prediction visualization

print(accuracy(y_test, y_test_pred))
print()
print(y_test)
print()
print(y_test_pred)

