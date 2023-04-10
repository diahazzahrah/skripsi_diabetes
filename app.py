from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier 
from neupy import algorithms 
from neupy.layers import *
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dataset_input")
def dataset_input():
    return render_template("dataset_input.html")


@app.route("/akurasi")
def akurasi():
    dataset = pd.read_csv("model/dataset/diabetes_after_preprocessing_all.csv")
    # ALL FEATURES
    Feature_variables_all = dataset.iloc[:, 0:8]
    Target_variable_all = dataset.iloc[:, 8].astype(int)
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
        Feature_variables_all,
        Target_variable_all,
        test_size=0.2,
        random_state=1,
    )
    acc_pnn_a = []
    acc_knn_a = []
    acc_pnn_e = []
    acc_knn_e = []
    # Accuracy
    classifier_pnn_a = algorithms.PNN(std=2, verbose=False).fit(
        X_train_a, y_train_a
    )

    kfold_pnn_a = KFold(
        n_splits=4, random_state=0, shuffle=True
    )  # k=10, split the data into 10 equal parts
    result_pnn_a = (
        cross_val_score(
            classifier_pnn_a,
            Feature_variables_all,
            Target_variable_all,
            cv=kfold_pnn_a,
            scoring="accuracy",
            n_jobs=1,
        )
        * 100
    )

    print(result_pnn_a * 100)
    print(
        "Hasil rata - rata Akurasi PNN -> K-Fold dengan All Features : ",
        round(result_pnn_a.mean() * 100, 1),
    )

    acc_pnn_a = round(result_pnn_a.mean(), 1)
    classifier_knn_a = KNeighborsClassifier(
        n_neighbors=11, metric="euclidean"
    ).fit(X_train_a, y_train_a)

    kfold_knn_a = KFold(
        n_splits=4, random_state=0, shuffle=True
    )  # k=10, split the data into 10 equal parts
    result_knn_a = (
        cross_val_score(
            classifier_knn_a,
            Feature_variables_all,
            Target_variable_all,
            cv=kfold_knn_a,
            scoring="accuracy",
            n_jobs=1,
        )
        * 100
    )

    print(result_knn_a * 100)
    print(
        "Hasil rata - rata Akurasi KNN -> K-Fold dengan All Features : ",
        round(result_knn_a.mean() * 100, 1),
    )
    acc_knn_a = round(result_knn_a.mean(), 1)

    # 4 FEATURES
    Feature_variables_empat = dataset.iloc[:, [0, 1, 5, 7]]
    Target_variable_empat = dataset.iloc[:, [8]].astype(int)
    X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
        Feature_variables_empat,
        Target_variable_empat,
        test_size=0.2,
        random_state=1,
    )
    classifier_pnn_e = algorithms.PNN(
        std=1, verbose=False, batch_size=128
    ).fit(X_train_e, y_train_e)
    kfold_pnn_e = KFold(
        n_splits=4, random_state=0, shuffle=True
    )  # k=10, split the data into 10 equal parts
    result_pnn_e = (
        cross_val_score(
            classifier_pnn_e,
            Feature_variables_empat,
            Target_variable_empat,
            cv=kfold_pnn_e,
            scoring="accuracy",
            n_jobs=1,
        )
        * 100
    )

    print(result_pnn_e * 100)
    print(
        "Hasil rata - rata Akurasi PNN -> K-Fold dengan 4 Features : ",
        round(result_pnn_e.mean() * 100, 1),
    )
    acc_pnn_e = round(result_pnn_e.mean(), 1)
    classifier_knn_e = KNeighborsClassifier(n_neighbors=13, metric="euclidean")
    classifier_knn_e.fit(X_train_e, y_train_e)

    kfold_knn_e = KFold(n_splits=4, random_state=0, shuffle=True)
    result_knn_e = (
        cross_val_score(
            classifier_knn_e,
            Feature_variables_empat,
            Target_variable_empat,
            cv=kfold_knn_e,
            scoring="accuracy",
            n_jobs=1,
        )
        * 100
    )

    print(result_knn_e * 100)
    print(
        "Hasil rata - rata Akurasi KNN -> K-Fold dengan 4 Features : ",
        round(result_knn_e.mean() * 100, 1),
    )
    acc_knn_e = round(result_knn_e.mean(), 1)
    return render_template(
        "akurasi.html",
        result_pnn_a=result_pnn_a,
        result_knn_a=result_knn_a,
        acc_pnn_a=acc_pnn_a,
        acc_knn_a=acc_knn_a,
        acc_knn_e=acc_knn_e,
        acc_pnn_e=acc_pnn_e,
        result_knn_e=result_knn_e,
        result_pnn_e=result_pnn_e,
    )

@app.route("/data_cleans")
def data_cleans():
    data_cleans = pd.read_csv(
        "model/dataset/Diabetes_DataCleaning.csv",
        error_bad_lines=False,
        delimiter=",",
        header=0,
    )
    data_cleans = np.array(data_cleans)
    pregnancies = data_cleans[:, 0]
    glucose = data_cleans[:, 1]
    bloodpressure = data_cleans[:, 2]
    skinthickness = data_cleans[:, 3]
    insulin = data_cleans[:, 4]
    bmi = data_cleans[:, 5]
    diabetespedigreefunction = data_cleans[:, 6]
    age = data_cleans[:, 7]
    outcome = data_cleans[:, 8]
    return render_template(
        "data_cleans.html",
        Pregnancies=pregnancies,
        Glucose=glucose,
        BloodPressure=bloodpressure,
        SkinThickness=skinthickness,
        Insulin=insulin,
        BMI=bmi,
        DiabetesPedigreeFunction=diabetespedigreefunction,
        Age=age,
        Outcome=outcome,
    )


@app.route("/dataset_preprocessing")
def dataset_preprocessing():
    dataset_preprocessing = pd.read_csv(
        "model/dataset/diabetes_after_preprocessing_all.csv",
        error_bad_lines=False,
        delimiter=",",
        header=0,
    )
    dataset_preprocessing = np.array(dataset_preprocessing)
    pregnancies = dataset_preprocessing[:, 0]
    glucose = dataset_preprocessing[:, 1]
    bloodpressure = dataset_preprocessing[:, 2]
    skinthickness = dataset_preprocessing[:, 3]
    insulin = dataset_preprocessing[:, 4]
    bmi = dataset_preprocessing[:, 5]
    diabetespedigreefunction = dataset_preprocessing[:, 6]
    age = dataset_preprocessing[:, 7]
    outcome = dataset_preprocessing[:, 8]
    return render_template(
        "dataset_preprocessing.html",
        Pregnancies=pregnancies,
        Glucose=glucose,
        BloodPressure=bloodpressure,
        SkinThickness=skinthickness,
        Insulin=insulin,
        BMI=bmi,
        DiabetesPedigreeFunction=diabetespedigreefunction,
        Age=age,
        Outcome=outcome,
    )


@app.route("/dataset_feature")
def dataset_feature():
    dataset_feature = pd.read_csv(
        "model/dataset/diabetes_after_preprocessing_4.csv",
        error_bad_lines=False,
        delimiter=",",
        header=0,
    )
    dataset_feature = np.array(dataset_feature)
    pregnancies = dataset_feature[:, 0]
    glucose = dataset_feature[:, 1]
    bmi = dataset_feature[:, 2]
    age = dataset_feature[:, 3]
    outcome = dataset_feature[:, 4]
    return render_template(
        "dataset_feature.html",
        Pregnancies=pregnancies,
        Glucose=glucose,
        BMI=bmi,
        Age=age,
        Outcome=outcome,
    )


@app.route("/result", methods=["GET", "POST"])
def result():
    if request.method == "GET":
        return render_template("output.html")
    elif request.method == "POST":
        print(dict(request.form))
        form_diabetes = dict(request.form)
        diabetes_features = form_diabetes.values()
        diabetes_features = np.array([float(x) for x in diabetes_features])
        classifier_knn1, scaler = joblib.load(
            "model/model-klasifikasi-diabetes.pkl"
        )
        diabetes_features = scaler.transform([diabetes_features])
        print(diabetes_features)
        result = classifier_knn1.predict(diabetes_features)
        outcome = {
            "0": "Not Diabetes",
            "1": "Diabetes",
        }
        result = outcome[str(result[0])]
        return render_template("output.html", result=result)
    else:
        return "Unsupported Request Method"


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/output")
def output():
    return render_template("output.html")


if __name__ == "__main__":
    app.debug = True
    app.run(port=5000)
