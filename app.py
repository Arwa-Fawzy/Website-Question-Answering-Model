from flask import Flask, render_template, request
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import nltk
import re

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

app = Flask(__name__)

def clean_and_extract_text(html):
    soup = BeautifulSoup(html, 'html.parser')
    texts = []
    for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li']):
        text = tag.get_text(strip=True)
        if text:
            texts.append(text)
    return " ".join(texts)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text.strip()

def chunk_text_sentences(text, max_tokens=400):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence.split()) <= max_tokens:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form.get("url")
        question = request.form.get("question")
        chunk_size = request.form.get("chunk-size", 400)

        if not url or not question:
            return render_template("index.html", error="من فضلك أدخل رابط الموقع والسؤال.")

        try:
            response = requests.get(url, timeout=10)
            raw_text = clean_and_extract_text(response.text)
            passage = clean_text(raw_text)

            if not passage:
                return render_template("index.html", error="لم يتم العثور على نص مناسب في الصفحة.")

        except Exception as e:
            return render_template("index.html", error=f"حدث خطأ أثناء تحميل الصفحة: {str(e)}")

        # Split text into chunks by sentence
        chunks = chunk_text_sentences(passage, max_tokens=int(chunk_size))

        # Load QA pipeline
        qa_model = pipeline("question-answering",
                            model="bert-large-uncased-whole-word-masking-finetuned-squad",
                            tokenizer="bert-large-uncased")

        best_answer = ""
        best_score = 0.0
        best_chunk = ""

        for chunk in chunks:
            try:
                result = qa_model(question=question, context=chunk)
                if result['score'] > best_score:
                    best_score = result['score']
                    best_answer = result['answer']
                    best_chunk = chunk
            except Exception as e:
                continue  # skip problematic chunk

        if best_answer:
            return render_template("index.html", answer=best_answer, highlighted_chunks=[best_chunk])
        else:
            return render_template("index.html", error="لم يتم العثور على إجابة واضحة للسؤال.")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
