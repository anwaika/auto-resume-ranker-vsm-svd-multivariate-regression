import os
import re
import nltk
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from docx import Document
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources (only runs first time)
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# -------- Extract Text --------
def extract_text(file_path):
    text = ""
    try:
        if file_path.endswith(".pdf"):
            reader = PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() or ""

        elif file_path.endswith(".docx"):
            doc = Document(file_path)
            for para in doc.paragraphs:
                text += para.text
    except Exception as e:
        print("Error reading file:", e)

    return text


# -------- Preprocess Text --------
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w not in stopwords.words('english')]
    return " ".join(filtered)


# -------- Main Route --------
@app.route("/", methods=["GET", "POST"])
def index():
    results = []

    if request.method == "POST":
        job_desc = request.form.get("job_description")
        files = request.files.getlist("resumes")

        if job_desc and files:

            job_desc = preprocess(job_desc)
            resume_texts = []
            file_names = []

            for file in files:
                if file.filename.endswith((".pdf", ".docx")):
                    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                    file.save(file_path)

                    text = extract_text(file_path)
                    processed = preprocess(text)

                    resume_texts.append(processed)
                    file_names.append(file.filename)

            if resume_texts:
                documents = [job_desc] + resume_texts

                vectorizer = TfidfVectorizer()
                vectors = vectorizer.fit_transform(documents)

                similarity_scores = cosine_similarity(
                    vectors[0:1], vectors[1:]
                ).flatten()

                scored = list(zip(file_names, similarity_scores))
                results = sorted(scored, key=lambda x: x[1], reverse=True)

    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)