# Task 2: FAQ Chatbot (ShopEase Online Store)

## What it does
A browser-based FAQ chatbot for an online store called ShopEase.
Users can type questions or select topics, and the chatbot responds 
with the most relevant answer using Natural Language Processing (NLP).

## Features
- 15 FAQs in 6 topics: Orders, Payments, Delivery, Returns & Refunds,
  Account and Support
- Topic buttons on the left to browse questions
- Clickable suggestion buttons and "Did you mean...?" suggestions
- Shows the match percentage for each answer
- Typing effect and a Clear chat button
- Typing animation effect
- Clear chat functionality
- Responsive web interface

## How it works
1. Text is cleaned with NLTK (lowercase, tokenize, remove stopwords
   and punctuation, lemmatize).
2. The FAQ questions are converted to TF-IDF vectors (scikit-learn).
3. Cosine similarity finds the FAQ closest to the user's question.
4. The matching answer is returned through the Python web server and displayed in the browser.

## How to run
1. Install Python 3.
2. Install libraries: `pip install -r requirements.txt`
3. Run: `python "FAQ Chatbot.py"`

The NLTK data downloads automatically the first time. Internet is
needed for that first run.

## Files
- FAQ Chatbot.py : main code
- Testing_FAQsChatbot.png : screenshot of the chatbot working
- requirements.txt : libraries needed

## Author
Umm e Rubab