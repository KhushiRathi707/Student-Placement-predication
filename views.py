from django.shortcuts import render

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import pandas as pd
import os


def index(request):
    return render(request, "index.html")


def predict(request):
    return render(request, "predict.html")


def result(request):

    # ==========================================
    # CSV FILE READ
    # ==========================================

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    csv_path = os.path.join(BASE_DIR, "project", "collegePlace.csv")
    dataframe = pd.read_csv(csv_path)

    # Remove extra spaces from column names
    dataframe.columns = dataframe.columns.str.strip()


    # ==========================================
    # GENDER CONVERSION
    # Male = 0
    # Female = 1
    # ==========================================

    dataframe["Gender"] = (
        dataframe["Gender"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "male": 0,
            "female": 1
        })
    )


    # ==========================================
    # STREAM CONVERSION
    # ==========================================

    dataframe["Stream"] = (
        dataframe["Stream"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "electronics and communication": 0,
            "computer science": 1,
            "information technology": 2,
            "mechanical": 3,
            "electrical": 4,
            "civil": 5
        })
    )


    # ==========================================
    # HOSTEL CONVERSION
    # Yes = 1
    # No = 0
    # ==========================================

    dataframe["Hostel"] = (
        dataframe["Hostel"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "yes": 1,
            "no": 0
        })
    )


    # ==========================================
    # HISTORY OF BACKLOGS CONVERSION
    # Yes = 1
    # No = 0
    # ==========================================

    dataframe["HistoryOfBacklogs"] = (
        dataframe["HistoryOfBacklogs"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "yes": 1,
            "no": 0
        })
    )


    # Remove invalid/missing rows
    dataframe = dataframe.dropna()


    # ==========================================
    # INPUT FEATURES
    # ==========================================

    X = dataframe[
        [
            "Age",
            "Gender",
            "Stream",
            "Internships",
            "CGPA",
            "Hostel",
            "HistoryOfBacklogs"
        ]
    ]


    # ==========================================
    # TARGET
    # ==========================================

    Y = dataframe["PlacedOrNot"]


    # ==========================================
    # TRAIN TEST SPLIT
    # ==========================================

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42
    )


    # ==========================================
    # LOGISTIC REGRESSION MODEL
    # ==========================================

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, Y_train)


    # ==========================================
    # GET USER INPUT
    # ==========================================

    try:

        # Age
        val1 = float(request.GET["n1"])


        # Gender
        gender = request.GET["n2"].strip().lower()

        if gender == "male":
            val2 = 0
        else:
            val2 = 1


        # Stream
        stream = request.GET["n3"].strip().lower()

        stream_dict = {
            "electronics and communication": 0,
            "computer science": 1,
            "information technology": 2,
            "mechanical": 3,
            "electrical": 4,
            "civil": 5
        }

        val3 = stream_dict.get(stream, 0)


        # Internships
        val4 = float(request.GET["n4"])


        # CGPA
        val5 = float(request.GET["n5"])


        # Hostel
        hostel = request.GET["n6"].strip().lower()

        if hostel == "yes":
            val6 = 1
        else:
            val6 = 0


        # History Of Backlogs
        backlog = request.GET["n7"].strip().lower()

        if backlog == "yes":
            val7 = 1
        else:
            val7 = 0


    except (ValueError, KeyError):

        return render(
            request,
            "predict.html",
            {
                "prediction": "Please enter valid values."
            }
        )


    # ==========================================
    # PREDICTION
    # ==========================================

    prediction = model.predict(
        [[
            val1,
            val2,
            val3,
            val4,
            val5,
            val6,
            val7
        ]]
    )


    # ==========================================
    # RESULT
    # ==========================================

    if prediction[0] == 1:

        result1 = "You are placed !!!!!"

    else:

        result1 = "You are not placed !!!!!"


    return render(
        request,
        "predict.html",
        {
            "prediction": result1
        }
    )