# ---------------------------------------------------
# This is not a runtime code, generates dependencies to infer intent i.e. 1) REQUIREMENT 2) BOQ 3) GENERAL
# shazri 2026
# ---------------------------------------------------

# -----------------------------
# ML-based Intent Classifier
# -----------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# -----------------------------
# Training data
# -----------------------------
data = [
    # -----------------------------
    # REQUIREMENTS queries
    # -----------------------------
    ("What are the requirements of the tender?", "REQUIREMENTS"),
    ("List the technical specifications for the tender", "REQUIREMENTS"),
    ("Show the project timelines and deadlines", "REQUIREMENTS"),
    ("Extract quality standards from the tender", "REQUIREMENTS"),
    ("Provide documents required for submission", "REQUIREMENTS"),
    ("What materials are required?", "REQUIREMENTS"),
    ("List mandatory spare parts and instruments", "REQUIREMENTS"),
    ("What quality standards apply?", "REQUIREMENTS"),
    ("Show all contract requirements", "REQUIREMENTS"),
    ("Give me the tender specifications", "REQUIREMENTS"),
    ("Provide the bid submission checklist", "REQUIREMENTS"),
    ("What are the mandatory technical specs?", "REQUIREMENTS"),
    ("Show the scope of work requirements", "REQUIREMENTS"),
    ("List deliverables and deadlines", "REQUIREMENTS"),
    ("Which documents must be submitted?", "REQUIREMENTS"),
    ("Give me all project requirements", "REQUIREMENTS"),
    ("Extract all specifications from the tender document", "REQUIREMENTS"),
    ("List all compliance standards to follow", "REQUIREMENTS"),
    ("What are the mandatory quality checkpoints?", "REQUIREMENTS"),
    ("Provide a summary of the required materials", "REQUIREMENTS"),
    ("Tell me about the requirements of tender", "REQUIREMENTS"),
    ("Tell me about the requirements", "REQUIREMENTS"),

    # -----------------------------
    # BOQ queries
    # -----------------------------
    ("Provide the Bill of Quantities", "BOQ"),
    ("Show the BoQ for this tender", "BOQ"),
    ("Extract Bill of Quantities from the document", "BOQ"),
    ("List quantities and pricing details", "BOQ"),
    ("Give me the cost breakdown", "BOQ"),
    ("Show itemized quantities and rates", "BOQ"),
    ("Provide the material quantity list", "BOQ"),
    ("Extract the pricing schedule", "BOQ"),
    ("List all items with quantities and prices", "BOQ"),
    ("Show the BOQ table", "BOQ"),
    ("Give me the quantity takeoff", "BOQ"),
    ("What are the unit rates and quantities?", "BOQ"),
    ("Provide detailed cost estimation items", "BOQ"),
    ("List construction quantities and costs", "BOQ"),
    ("Extract all measurable items and quantities", "BOQ"),

    # -----------------------------
    # GENERAL queries
    # -----------------------------
    ("Tell me about tender", "GENERAL"),
    ("Explain the tender process", "GENERAL"),
    ("Who is the issuing authority?", "GENERAL"),
    ("How can I submit my bid?", "GENERAL"),
    ("When will the results be announced?", "GENERAL"),
    ("What is the eligibility criteria?", "GENERAL"),
    ("How long does the evaluation take?", "GENERAL"),
    ("Give an overview of the project", "GENERAL"),
    ("Who are the stakeholders?", "GENERAL"),
    ("Provide background about the project", "GENERAL"),
    ("What is the purpose of this tender?", "GENERAL"),
    ("Explain the bidding procedure", "GENERAL"),
    ("How are bids evaluated?", "GENERAL"),
    ("Where can I find tender updates?", "GENERAL"),
    ("What is the timeline for announcements?", "GENERAL"),
    ("Who can participate in this tender?", "GENERAL"),
    ("Give me the general project overview", "GENERAL"),
    ("Explain eligibility requirements", "GENERAL"),
    ("What is the overall project scope?", "GENERAL"),
    ("Tell me about the tender background", "GENERAL"),
]

queries = [x[0] for x in data]
labels = [x[1] for x in data]

# -----------------------------
# Vectorize queries
# -----------------------------
vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
X = vectorizer.fit_transform(queries)

# -----------------------------
# Train classifier
# -----------------------------
clf = LogisticRegression(max_iter=500)
clf.fit(X, labels)

# -----------------------------
# Prediction function
# -----------------------------
def predict_intent(query):
    vec = vectorizer.transform([query])
    return clf.predict(vec)[0]

# -----------------------------
# Test examples
# -----------------------------
test_queries = [
    "What are the requirements of tender?",
    "Tell me about tender",
    "Show me the quality standards",
    "Explain the tender submission process",
    "List mandatory documents for the bid",
    "Who is issuing this tender?",
    "Provide all contract specifications",

    # BOQ tests
    "Show me the bill of quantities",
    "Give cost breakdown of materials",
    "List quantities and unit rates",
]

for q in test_queries:
    print(f"Query: {q}\nPredicted intent: {predict_intent(q)}\n")

# -----------------------------
# Save model for later use
# -----------------------------
joblib.dump(clf, "intent_clf.pkl")
joblib.dump(vectorizer, "intent_vectorizer.pkl")

print("Model saved successfully.")