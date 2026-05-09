from flask import Flask, request, redirect, session, jsonify, render_template
from datetime import datetime
from datetime import timedelta
import json
import os
import sys

try:
    import mysql.connector
    from mysql.connector import Error
except ImportError:
    mysql = None
    Error = Exception

# ── Auto-find project folder ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ── Import prediction model ──
from model import train_model, predict_disease, get_all_symptoms

# ── Auto-find HTML folder ──
def find_file(filename):
    for root, dirs, files in os.walk(BASE_DIR):
        if filename in files:
            return root
    return BASE_DIR

HTML_DIR = find_file("index.html")
print(f"✅ HTML folder: {HTML_DIR}")

TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.secret_key = "secret123"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(minutes=5)

history = {}
BASE_DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", "T#ahdf342"),
}
DB_NAME = os.getenv("MYSQL_DATABASE", "health_management")
USERS_JSON_PATH = os.path.join(BASE_DIR, "users.json")
POINT_RULES = {
    "water": 10,
    "exercise": 10,
    "sleep": 10,
    "diet": 10,
    "meditate": 10,
}
DOCTORS = [
    {"name": "Dr. Meera Kulkarni", "specialty": "General Physician", "city": "Nagpur", "hospital": "CarePlus Clinic", "distance": "1.8 km", "rating": 4.8, "availability": "Mon-Sat, 10 AM - 6 PM", "phone": "+91 98765 11111"},
    {"name": "Dr. Arjun Deshmukh", "specialty": "Cardiologist", "city": "Nagpur", "hospital": "Heartbeat Multispeciality", "distance": "3.2 km", "rating": 4.7, "availability": "Mon-Fri, 11 AM - 5 PM", "phone": "+91 98765 22222"},
    {"name": "Dr. Sana Qureshi", "specialty": "Dermatologist", "city": "Pune", "hospital": "SkinSense Center", "distance": "2.5 km", "rating": 4.9, "availability": "Tue-Sun, 9 AM - 3 PM", "phone": "+91 98765 33333"},
    {"name": "Dr. Rohan Patil", "specialty": "Pediatrician", "city": "Mumbai", "hospital": "LittleCare Hospital", "distance": "4.0 km", "rating": 4.6, "availability": "Mon-Sat, 8 AM - 2 PM", "phone": "+91 98765 44444"},
    {"name": "Dr. Neha Joshi", "specialty": "Gynecologist", "city": "Nashik", "hospital": "Women Wellness Hub", "distance": "2.1 km", "rating": 4.8, "availability": "Mon-Fri, 10 AM - 4 PM", "phone": "+91 98765 55555"},
    {"name": "Dr. Vikram Shah", "specialty": "Orthopedic", "city": "Nagpur", "hospital": "MoveWell Ortho Care", "distance": "5.4 km", "rating": 4.5, "availability": "Mon-Sat, 1 PM - 8 PM", "phone": "+91 98765 66666"},
]

# ── Read HTML files directly ──
def read_html(filename):
    filepath = os.path.join(HTML_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def ensure_mysql_available():
    if mysql is None:
        raise RuntimeError(
            "MySQL connector is not installed. Run: pip install mysql-connector-python"
        )


def get_server_connection():
    ensure_mysql_available()
    return mysql.connector.connect(**BASE_DB_CONFIG)


def get_db_connection():
    ensure_mysql_available()
    return mysql.connector.connect(database=DB_NAME, **BASE_DB_CONFIG)


def init_db():
    server_conn = None
    db_conn = None
    try:
        server_conn = get_server_connection()
        with server_conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}`")
        server_conn.commit()

        db_conn = get_db_connection()
        with db_conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    age INT NULL,
                    gender VARCHAR(20) NULL,
                    blood_group VARCHAR(10) NULL,
                    email VARCHAR(255) NOT NULL UNIQUE,
                    password VARCHAR(255) NOT NULL,
                    user_points INT NOT NULL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute("SHOW COLUMNS FROM users LIKE 'user_points'")
            if cursor.fetchone() is None:
                cursor.execute(
                    """
                    ALTER TABLE users
                    ADD COLUMN user_points INT NOT NULL DEFAULT 0
                    """
                )
        db_conn.commit()
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()
        if server_conn and server_conn.is_connected():
            server_conn.close()


def migrate_users_from_json():
    if not os.path.exists(USERS_JSON_PATH):
        return

    with open(USERS_JSON_PATH, "r", encoding="utf-8") as file:
        users = json.load(file)

    if not users:
        return

    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            for email, user in users.items():
                age_value = user.get("age")
                age = int(age_value) if str(age_value).strip().isdigit() else None
                cursor.execute(
                    """
                    INSERT INTO users (name, age, gender, blood_group, email, password)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        name = VALUES(name),
                        age = VALUES(age),
                        gender = VALUES(gender),
                        blood_group = VALUES(blood_group),
                        password = VALUES(password)
                    """,
                    (
                        user.get("name", "User"),
                        age,
                        user.get("gender"),
                        user.get("blood"),
                        email,
                        user.get("password", ""),
                    ),
                )
        conn.commit()
    finally:
        if conn and conn.is_connected():
            conn.close()


def get_user_by_email(email):
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
            return cursor.fetchone()
    finally:
        if conn and conn.is_connected():
            conn.close()


def create_user(name, age, gender, blood_group, email, password):
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO users (name, age, gender, blood_group, email, password)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (name, age, gender, blood_group, email, password),
            )
        conn.commit()
        return True
    except Error as exc:
        if conn:
            conn.rollback()
        if getattr(exc, "errno", None) == 1062:
            return False
        raise
    finally:
        if conn and conn.is_connected():
            conn.close()


def get_user_points(email):
    user = get_user_by_email(email)
    if not user:
        return 0
    return user.get("user_points", 0)


def add_points(email, points):
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE users SET user_points = user_points + %s WHERE email = %s",
                (points, email),
            )
        conn.commit()
    finally:
        if conn and conn.is_connected():
            conn.close()


def get_leaderboard():
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT email, user_points
                FROM users
                ORDER BY user_points DESC, created_at ASC
                LIMIT 10
                """
            )
            return cursor.fetchall()
    finally:
        if conn and conn.is_connected():
            conn.close()


def filter_doctors(city="", specialty="", query=""):
    city = city.strip().lower()
    specialty = specialty.strip().lower()
    query = query.strip().lower()
    results = []

    for doctor in DOCTORS:
        if city and doctor["city"].lower() != city:
            continue
        if specialty and specialty not in doctor["specialty"].lower():
            continue
        if query:
            searchable = " ".join(
                [doctor["name"], doctor["specialty"], doctor["hospital"], doctor["city"]]
            ).lower()
            if query not in searchable:
                continue
        results.append(doctor)

    return results


try:
    init_db()
    migrate_users_from_json()
except Exception as exc:
    print(f"MySQL setup warning: {exc}")

try:
    MODEL, FEATURE_NAMES = train_model(os.path.join(BASE_DIR, "disease_dataset_1000.xlsx"))
except Exception as exc:
    MODEL, FEATURE_NAMES = None, []
    print(f"Model setup warning: {exc}")

# ════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════

@app.route("/")
def home():
    return redirect("/login")


@app.route("/home")
def dashboard_home():
    if "user" not in session:
        return redirect("/login")
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip()
        password = request.form["password"]
        user = get_user_by_email(email)
        if user and user["password"] == password:
            session.permanent = True
            session["user"] = email
            return redirect("/home")
        return render_template("login.html", error="Invalid email or password.")
    return render_template("login.html", error=None)

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form["email"].strip()
        age_value = request.form.get("age", "").strip()
        age = int(age_value) if age_value.isdigit() else None
        created = create_user(
            request.form.get("name", "User").strip() or "User",
            age,
            request.form.get("gender", "").strip() or None,
            request.form.get("blood", "").strip() or None,
            email,
            request.form["password"],
        )
        if created:
            history[email] = []
            return redirect("/login")
        return "User with this email already exists.", 400
    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/login")

@app.route("/profile")
def profile():
    if "user" not in session:
        return redirect("/login")
    return render_template("profile.html", user=get_user_by_email(session["user"]))

@app.route("/disease-history")
def disease_history():
    if "user" not in session:
        return redirect("/login")
    return render_template(
        "diseasehistory.html",
        history_records=history.get(session["user"], []),
    )


@app.route("/games", methods=["GET", "POST"])
def games():
    if "user" not in session:
        return redirect("/login")

    earned_points = 0
    if request.method == "POST":
        tasks = request.form.getlist("tasks")
        earned_points = sum(POINT_RULES.get(task, 0) for task in tasks)
        if earned_points:
            add_points(session["user"], earned_points)

    return render_template(
        "games.html",
        user_points=get_user_points(session["user"]),
        earned_points=earned_points,
    )


@app.route("/leaderboard")
def leaderboard():
    if "user" not in session:
        return redirect("/login")
    return render_template("leaderboard.html", leaderboard=get_leaderboard())


@app.route("/blogs")
def blogs():
    return render_template("blogs.html")


@app.route("/nearby-doctors")
def nearby_doctors():
    city = request.args.get("city", "").strip()
    specialty = request.args.get("specialty", "").strip()
    query = request.args.get("query", "").strip()
    return render_template(
        "doctors.html",
        doctors=filter_doctors(city=city, specialty=specialty, query=query),
        cities=sorted({doctor["city"] for doctor in DOCTORS}),
        specialties=sorted({doctor["specialty"] for doctor in DOCTORS}),
        selected_city=city,
        selected_specialty=specialty,
        query=query,
    )

# ════════════════════════════════════════
#  PREDICT — Main ML route
# ════════════════════════════════════════
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    # Collect up to 5 symptoms from form
    symptoms = []
    for i in range(1, 6):
        s = request.form.get(f"symptom{i}", "").strip()
        if s:
            symptoms.append(s)

    # Also support comma-separated input
    raw = request.form.get("symptoms", "")
    if raw:
        symptoms += [s.strip() for s in raw.split(",") if s.strip()]

    if not symptoms:
        return redirect("/predict")

    if MODEL is None or not FEATURE_NAMES:
        return "Prediction model is not available right now.", 500

    # Run model
    results, matched = predict_disease(MODEL, FEATURE_NAMES, symptoms)

    if not results:
        return f"""
        <h2>No matching disease found.</h2>
        <p>Symptoms entered: {symptoms}</p>
        <p>Please check symptom spelling. <a href='/predict'>Try again</a></p>
        """

    top = results[0]  # Best prediction

    # Save to history
    if "user" in session:
        history.setdefault(session["user"], []).append({
            "date":        datetime.now().strftime("%Y-%m-%d %H:%M"),
            "symptoms":    ", ".join(matched),
            "disease":     top["disease"],
            "confidence":  top["confidence"],
            "precautions": ", ".join(top["precautions"])
        })

    # Build result HTML
    other_html = ""
    for r in results[1:]:
        other_html += f"""
        <div style="background:rgba(255,255,255,0.3);border-radius:12px;padding:14px 18px;margin-bottom:10px;border:1px solid rgba(107,143,214,0.2)">
            <b>{r['disease']}</b>
            <span style="float:right;color:#4a6bbf;font-weight:600">{r['confidence']}%</span>
        </div>"""

    precaution_items = "".join(
        f"<li style='padding:6px 0'>✅ {p}</li>" for p in top["precautions"]
    )
    matched_tags = "".join(
        f"<span style='background:rgba(107,143,214,0.15);border-radius:50px;padding:4px 14px;margin:4px;display:inline-block;font-size:0.85rem'>{s.replace('_',' ')}</span>"
        for s in matched
    )

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Prediction Result</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{
                font-family: 'DM Sans', sans-serif;
                min-height: 100vh;
                background: linear-gradient(135deg, #e8edfb 0%, #c5d4f5 100%);
                padding: 40px 16px;
            }}
            .card {{
                background: rgba(255,255,255,0.55);
                backdrop-filter: blur(18px);
                border: 1px solid rgba(255,255,255,0.6);
                border-radius: 20px;
                box-shadow: 0 8px 32px rgba(107,143,214,0.15);
                max-width: 680px;
                margin: 0 auto;
                padding: 40px;
            }}
            .disease-name {{
                font-size: 2rem;
                font-weight: 700;
                color: #1a2e5a;
            }}
            .confidence-badge {{
                background: linear-gradient(135deg, #8faee0, #4a6bbf);
                color: white;
                border-radius: 50px;
                padding: 6px 20px;
                font-weight: 600;
                font-size: 1rem;
            }}
            .btn-back {{
                background: linear-gradient(135deg, #8faee0, #4a6bbf);
                color: white;
                border: none;
                border-radius: 12px;
                padding: 12px 32px;
                font-weight: 600;
                cursor: pointer;
                text-decoration: none;
                display: inline-block;
                margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="card">
            <div style="text-align:center;margin-bottom:28px">
                <div style="font-size:3rem">🩺</div>
                <h2 style="font-size:1rem;color:#6b8fb8;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:6px">Top Prediction</h2>
                <div class="disease-name">{top['disease']}</div>
                <span class="confidence-badge">{top['confidence']}% confidence</span>
            </div>

            <h5 style="color:#1a2e5a;margin-bottom:10px">📌 Matched Symptoms</h5>
            <div style="margin-bottom:24px">{matched_tags if matched_tags else '<p style="color:#6b8fb8">No exact matches — try checking spelling</p>'}</div>

            <h5 style="color:#1a2e5a;margin-bottom:10px">💊 Recommended Precautions</h5>
            <ul style="padding-left:20px;color:#3e3d3d;margin-bottom:24px">
                {precaution_items}
            </ul>

            <h5 style="color:#1a2e5a;margin-bottom:12px">🔍 Other Possibilities</h5>
            {other_html}

            <div style="text-align:center">
                <a href="/predict" class="btn-back">← Try Again</a>
                <a href="/home" class="btn-back" style="margin-left:10px">🏠 Home</a>
            </div>
        </div>
    </body>
    </html>
    """

# ── API: get all symptoms for autocomplete ──
@app.route("/api/symptoms")
def api_symptoms():
    return jsonify(get_all_symptoms(FEATURE_NAMES))

# ── API: get user history ──
@app.route("/api/history")
def api_history():
    if "user" not in session:
        return jsonify([])
    return jsonify(history.get(session["user"], []))

if __name__ == "__main__":
    print("🚀 Starting Health Care app...")
    app.run(debug=True)
