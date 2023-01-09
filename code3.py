from sklearn.metrics import accuracy_score

from code1 import X_Data, Y_Data
from code2 import models

for name, model in models:
    model.fit(X_Data, Y_Data.values.ravel())
    Y_pred = model.predict(X_Data)
    print(name, "'s Accuracy is ", accuracy_score(Y_Data, Y_pred))
