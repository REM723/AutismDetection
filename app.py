import pickle
import warnings

from flask import Flask, jsonify, request, send_from_directory

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__, static_folder=".", static_url_path="")

with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# ponytail: same one-off renames the training notebook applied before encoding
COUNTRY_MAP = {"Viet Nam": "Vietnam", "AmericanSamoa": "United States", "Hong Kong": "China"}
ETHNICITY_MAP = {"?": "Others", "others": "Others"}
RELATION_MAP = {"?": "Others", "Relative": "Others", "Parent": "Others", "Health care professional": "Others"}

FEATURE_ORDER = [
    "A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score", "A6_Score",
    "A7_Score", "A8_Score", "A9_Score", "A10_Score", "age", "gender",
    "ethnicity", "jaundice", "austim", "contry_of_res", "used_app_before",
    "result", "relation",
]


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/options")
def options():
    return jsonify({col: list(enc.classes_) for col, enc in encoders.items()})


@app.route("/predict", methods=["POST"])
def predict():
    row = request.get_json()

    row["contry_of_res"] = COUNTRY_MAP.get(row["contry_of_res"], row["contry_of_res"])
    row["ethnicity"] = ETHNICITY_MAP.get(row["ethnicity"], row["ethnicity"])
    row["relation"] = RELATION_MAP.get(row["relation"], row["relation"])

    for col, enc in encoders.items():
        if row[col] not in enc.classes_:
            return jsonify({"error": f"Unrecognized value for {col}: {row[col]}"}), 400
        row[col] = int(enc.transform([row[col]])[0])

    features = [[row[f] for f in FEATURE_ORDER]]
    prediction = int(model.predict(features)[0])
    proba_raw = model.predict_proba(features)[0]
    # ponytail: model was pickled under sklearn 1.3.2; current sklearn's predict_proba
    # doesn't renormalize tree vote counts for that version, so re-normalize here.
    probability = float(proba_raw[1] / proba_raw.sum())
    return jsonify({"prediction": prediction, "probability": probability})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
