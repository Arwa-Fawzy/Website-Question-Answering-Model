# Intelligent Question Answering System 

This project implements a BERT-based extractive Question Answering (QA) model on content scraped from the Arabic news website موقع مصراوى. It allows users to ask questions and receive accurate answers directly from the site's content.

---

## Project Overview

The system is designed to extract and highlight precise answers from articles on موقع مصراوى using state-of-the-art NLP techniques, particularly a fine-tuned BERT model. It includes web scraping, data preprocessing, QA modeling, and a simple API using Flask.

---

## Implementation Steps

### 1. Data Collection
- Web Scraping: Scrape pages from موقع مصراوى using tools like `BeautifulSoup`, `Scrapy`, or `Selenium`.
- HTML Parsing: Extract main content from tags such as `<p>`, `<h1-h6>`, `<ul>`, and remove scripts, ads, and irrelevant tags.

### 2. Preprocessing
- Text Cleaning: Remove HTML tags, special characters, and redundant spaces.
- Text Chunking: Split long article texts into smaller, manageable chunks.
- Metadata Extraction: Store additional context like article titles, section headers, and source URLs.

### 3. Model Selection
- Model Used: Fine-tuned BERT model for Extractive QA (e.g., `bert-base-uncased`, `bert-base-arabic`, or similar), trained on datasets like SQuAD.
- Objective: Identify and return the most relevant span from the input text that answers the user’s question.

### 4. Search & Retrieval
- Retrieval Technique: Use BM25 or dense vector retrieval (e.g., DPR) to fetch top relevant text chunks based on the question.

### 5. Answer Generation
- Approach: Pass retrieved text and user question to the BERT QA model.
- Output: Return the extracted answer span from the relevant content.

### 6. Deployment
- Framework: Deployed using Flask.
- Integration: Can be connected to web interfaces or messaging platforms (optional).
- Deliverable: Students must submit API documentation with endpoint definitions and usage examples.

---
