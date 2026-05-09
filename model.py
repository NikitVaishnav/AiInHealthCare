import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ─── Precautions per disease ───
PRECAUTIONS = {
    "Fungal infection":         ["Keep skin dry", "Use antifungal cream", "Avoid sharing personal items", "Wear breathable clothing"],
    "Allergy":                  ["Avoid allergens", "Take antihistamines", "Keep environment clean", "Consult allergist"],
    "GERD":                     ["Avoid spicy food", "Eat small meals", "Don't lie down after eating", "Take antacids"],
    "Chronic cholestasis":      ["Avoid fatty foods", "Take prescribed medication", "Stay hydrated", "Regular liver checkups"],
    "Drug Reaction":            ["Stop the drug immediately", "Consult your doctor", "Take antihistamines", "Monitor symptoms"],
    "Peptic ulcer disease":     ["Avoid NSAIDs", "Eat bland food", "Take antacids", "Reduce stress"],
    "AIDS":                     ["Use protection", "Take antiretroviral therapy", "Regular checkups", "Maintain hygiene"],
    "Diabetes":                 ["Monitor blood sugar", "Follow diet plan", "Exercise regularly", "Take insulin if prescribed"],
    "Gastroenteritis":          ["Stay hydrated", "Eat bland food", "Rest", "Avoid dairy products"],
    "Bronchial Asthma":         ["Use inhaler", "Avoid triggers", "Stay away from smoke", "Consult pulmonologist"],
    "Hypertension":             ["Reduce salt intake", "Exercise regularly", "Take prescribed medication", "Reduce stress"],
    "Migraine":                 ["Rest in dark room", "Take pain relievers", "Stay hydrated", "Avoid triggers"],
    "Cervical spondylosis":     ["Do neck exercises", "Use cervical pillow", "Avoid heavy lifting", "Consult physiotherapist"],
    "Paralysis (brain hemorrhage)": ["Immediate hospitalization", "Physical therapy", "Follow doctor's advice", "Monitor vitals"],
    "Jaundice":                 ["Rest", "Stay hydrated", "Avoid alcohol", "Eat light meals"],
    "Malaria":                  ["Take antimalarial drugs", "Use mosquito nets", "Avoid stagnant water", "Stay indoors at dusk"],
    "Chicken pox":              ["Avoid scratching", "Take antivirals", "Stay isolated", "Use calamine lotion"],
    "Dengue":                   ["Stay hydrated", "Take paracetamol", "Avoid aspirin", "Rest"],
    "Typhoid":                  ["Take antibiotics", "Drink clean water", "Eat cooked food", "Rest"],
    "Hepatitis A":              ["Rest", "Avoid alcohol", "Stay hydrated", "Eat light meals"],
    "Hepatitis B":              ["Take antiviral medication", "Avoid alcohol", "Rest", "Regular liver tests"],
    "Hepatitis C":              ["Take antiviral medication", "Avoid alcohol", "Regular checkups", "Healthy diet"],
    "Hepatitis D":              ["Take prescribed medication", "Avoid alcohol", "Rest", "Regular liver tests"],
    "Hepatitis E":              ["Rest", "Stay hydrated", "Avoid alcohol", "Eat light meals"],
    "Alcoholic hepatitis":      ["Stop alcohol completely", "Take prescribed medication", "Healthy diet", "Rest"],
    "Tuberculosis":             ["Complete antibiotic course", "Cover mouth while coughing", "Stay isolated", "Eat nutritious food"],
    "Common Cold":              ["Rest", "Drink fluids", "Take decongestants", "Gargle with salt water"],
    "Pneumonia":                ["Take antibiotics", "Rest", "Stay hydrated", "Seek hospitalization if severe"],
    "Dimorphic hemorrhoids":    ["Eat high-fiber diet", "Stay hydrated", "Avoid straining", "Use sitz bath"],
    "Heart attack":             ["Call emergency immediately", "Chew aspirin", "Rest", "CPR if needed"],
    "Varicose veins":           ["Elevate legs", "Wear compression stockings", "Exercise regularly", "Avoid long standing"],
    "Hypothyroidism":           ["Take thyroid medication", "Exercise regularly", "Eat iodine-rich food", "Regular checkups"],
    "Hyperthyroidism":          ["Take antithyroid drugs", "Avoid iodine-rich food", "Rest", "Regular checkups"],
    "Hypoglycemia":             ["Eat sugar or candy", "Drink fruit juice", "Monitor blood sugar", "Consult doctor"],
    "Osteoarthritis":           ["Exercise gently", "Use pain relievers", "Maintain healthy weight", "Physiotherapy"],
    "Arthritis":                ["Exercise regularly", "Take anti-inflammatory drugs", "Apply hot/cold packs", "Rest joints"],
    "Vertigo":                  ["Rest", "Avoid sudden movements", "Take prescribed medication", "Head exercises"],
    "Acne":                     ["Keep skin clean", "Use non-comedogenic products", "Avoid touching face", "Consult dermatologist"],
    "Urinary tract infection":  ["Drink plenty of water", "Take antibiotics", "Avoid caffeine", "Maintain hygiene"],
    "Psoriasis":                ["Moisturize skin", "Use prescribed creams", "Avoid triggers", "Consult dermatologist"],
    "Impetigo":                 ["Take antibiotics", "Keep wounds clean", "Avoid touching sores", "Wash hands frequently"],
}
SEVERE_DISEASES = {
    "Tuberculosis",
    "Heart attack",
    "AIDS",
    "Paralysis (brain hemorrhage)",
    "Hepatitis B",
    "Hepatitis C",
    "Hepatitis D",
    "Alcoholic hepatitis",
    "Pneumonia",
}
SEVERE_DISEASE_REQUIRED_SYMPTOMS = {
    "Tuberculosis": {"cough", "high_fever", "weight_loss", "loss_of_appetite", "chest_pain", "breathlessness", "fatigue", "sweating"},
    "Pneumonia": {"cough", "high_fever", "breathlessness", "chest_pain"},
    "Heart attack": {"chest_pain", "breathlessness", "sweating", "vomiting"},
}

# ─── Load dataset ───
def normalize_symptom(symptom):
    return symptom.strip().lower().replace(" ", "_")


def load_dataset(path="disease_dataset_1000.xlsx"):
    df = pd.read_excel(path, sheet_name="Dataset")
    if "Sample_ID" in df.columns:
        df = df.drop(columns=["Sample_ID"])
    X = df.drop(columns=["Disease"]).values
    y = df["Disease"].values
    feature_names = list(df.drop(columns=["Disease"]).columns)
    return df, X, y, feature_names


def build_disease_profiles(df, feature_names):
    grouped = df.groupby("Disease")[feature_names].mean()
    return {
        disease: grouped.loc[disease].to_numpy(dtype=float)
        for disease in grouped.index
    }


def apply_severity_bias(disease, score, matched_strength, overlap_ratio):
    if disease not in SEVERE_DISEASES:
        return score

    # Severe diseases should appear only when the entered symptoms strongly fit.
    if matched_strength < 0.55 and overlap_ratio < 0.35:
        return score * 0.2
    if matched_strength < 0.7 and overlap_ratio < 0.5:
        return score * 0.45
    return score


def apply_symptom_gate(disease, score, matched_symptoms):
    required_pool = SEVERE_DISEASE_REQUIRED_SYMPTOMS.get(disease)
    if not required_pool:
        return score

    matched_required = len(set(matched_symptoms) & required_pool)
    if matched_required == 0:
        return score * 0.05
    if matched_required == 1 and len(matched_symptoms) < 3:
        return score * 0.15
    return score

# ─── Train model with train/test split ───
def train_model(path="disease_dataset_1000.xlsx", test_size=0.2, random_state=42):
    df, X, y, feature_names = load_dataset(path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=400,
        random_state=random_state,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return {
        "classifier": model,
        "feature_names": feature_names,
        "disease_profiles": build_disease_profiles(df, feature_names),
    }, feature_names

# ─── Predict ───
def predict_disease(model, feature_names, user_symptoms):
    """
    user_symptoms: list of symptom strings entered by user
    returns: (results, matched_symptoms)
    """
    cleaned = []
    for symptom in user_symptoms:
        if symptom.strip():
            normalized = normalize_symptom(symptom)
            if normalized not in cleaned:
                cleaned.append(normalized)

    row = [0] * len(feature_names)
    matched = []
    matched_indexes = []
    for s in cleaned:
        if s in feature_names:
            index = feature_names.index(s)
            row[index] = 1
            matched.append(s)
            matched_indexes.append(index)

    if not matched:
        return None, []

    classifier = model["classifier"] if isinstance(model, dict) else model
    disease_profiles = model.get("disease_profiles", {}) if isinstance(model, dict) else {}
    rf_probs = classifier.predict_proba([row])[0]
    rf_scores = dict(zip(classifier.classes_, rf_probs))
    row_array = np.array(row, dtype=float)
    input_count = max(len(matched_indexes), 1)
    ranked = []

    for disease, rf_score in rf_scores.items():
        profile = disease_profiles.get(disease)
        profile_score = 0.0
        matched_strength = 0.0
        overlap_ratio = 0.0

        if profile is not None and matched_indexes:
            matched_strength = float(np.mean(profile[matched_indexes]))
            expected_strength = float(np.sum(profile * row_array) / input_count)
            active_symptom_count = max(int(np.sum(profile >= 0.5)), 1)
            overlap_count = int(np.sum((profile >= 0.5) & (row_array == 1)))
            overlap_ratio = overlap_count / active_symptom_count
            profile_score = (0.5 * matched_strength) + (0.3 * expected_strength) + (0.2 * overlap_ratio)

        final_score = (0.75 * profile_score) + (0.25 * float(rf_score))
        final_score = apply_severity_bias(
            disease,
            final_score,
            matched_strength,
            overlap_ratio,
        )
        final_score = apply_symptom_gate(disease, final_score, matched)
        ranked.append((disease, final_score))

    ranked.sort(key=lambda item: item[1], reverse=True)

    results = []
    for disease, score in ranked[:3]:
        confidence = round(score * 100, 1)
        precautions = PRECAUTIONS.get(disease, ["Consult a doctor"])
        results.append({
            "disease": disease,
            "confidence": confidence,
            "precautions": precautions,
        })

    return results, matched

def get_all_symptoms(feature_names):
    return sorted(feature_names)


if __name__ == "__main__":
    model, feature_names = train_model("disease_dataset_1000.xlsx")

    # Quick prediction test
    test_symptoms = ["itching", "skin_rash", "fatigue", "high_fever", "blister"]
    results, matched = predict_disease(model, feature_names, test_symptoms)

    print("\n--- Prediction Test ---")
    print("Input symptoms:", test_symptoms)
    print("Matched symptoms:", matched)
    for r in results:
        print(f"\nDisease: {r['disease']} ({r['confidence']}%)")
        print("Precautions:", r['precautions'])
