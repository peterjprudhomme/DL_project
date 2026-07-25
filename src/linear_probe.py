from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


class LinearProbe:

    def __init__(self):

        self.model = LogisticRegression(
            max_iter=1000
        )

    def fit(
        self,
        X,
        y,
        test_size=0.2,
        random_state=42
    ):

        (
            X_train,
            X_test,
            y_train,
            y_test

        ) = train_test_split(

            X,
            y,

            test_size=test_size,

            random_state=random_state,

            stratify=y
        )

        self.model.fit(
            X_train,
            y_train
        )

        predictions = self.model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        report = classification_report(
            y_test,
            predictions
        )

        matrix = confusion_matrix(
            y_test,
            predictions
        )

        return {

            "accuracy": accuracy,

            "report": report,

            "confusion_matrix": matrix
        }

    def predict(
        self,
        X
    ):

        return self.model.predict(X)

    def coefficients(self):

        return self.model.coef_[0]