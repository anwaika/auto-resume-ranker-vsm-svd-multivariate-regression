# Auto Resume Ranker using Vector Space Models, Singular Value Decomposition, and Multivariate Regression

This project implements an intelligent resume ranking system that compares candidate resumes with a job description and ranks them based on relevance.

The system uses mathematical and machine learning concepts including:

• Vector Space Models (TF-IDF representation)  
• Singular Value Decomposition (SVD) for dimensionality reduction  
• Cosine Similarity for document comparison  
• Multivariate Regression concepts for ranking interpretation

## System Workflow

1.User inputs job description and uploads resumes (PDF / DOCX / TXT)
2.Text is extracted from each resume
3.Text is preprocessed (lowercased, tokenized)
4.TF-IDF converts text to vectors → SVD reduces dimensions → Cosine Similarity computed
5.Sentence Transformer encodes job description and each resume → semantic similarity computed
6.Skill score and years of experience extracted as additional features
7.Ridge Regression combines all features into a final weighted score
8.Resumes ranked and displayed with score breakdown

## Mathematical Concepts Used

Vector Space Models  
Singular Value Decomposition (SVD)  
Multivariate Regression Concepts  
Cosine Similarity

# AI Component

Sentence Transformers (all-MiniLM-L6-v2) — pretrained transformer model that encodes text into semantic embeddings, enabling contextual understanding beyond keyword matching

# Tech Stack

Backend: Flask (Python)
NLP: NLTK, scikit-learn (TF-IDF), sentence-transformers
Math/ML: NumPy, scikit-learn (SVD, Ridge Regression, Cosine Similarity)
File Parsing: PyPDF2, python-docx