# FUTURE_ML_02 – Support Ticket Classification System

**Task 2** of the **Future Interns Machine Learning Internship**  
Automated classification & priority assignment for customer support tickets using NLP

## Project Overview

This project builds an **NLP-based Support Ticket Classification system** that:

- Classifies customer support tickets into categories: **Account**, **Billing**, **Technical**, **General**  
- Assigns priority levels: **High** 🚨, **Medium** ⚠️, **Low** ℹ️  
- Helps support teams triage tickets faster and prioritize urgent issues  


## Features

- Text preprocessing (lowercase, remove punctuation, custom stopwords)  
- TF-IDF vectorization with bigrams  
- Logistic Regression (category) + Multinomial Naive Bayes (priority)  
- Stratified train-test split & class balancing  
- Confusion matrix visualization  
- Interactive **Streamlit** web app with:
  - Emoji-enhanced UI  
  - Colored result cards  
  - Priority-based alerts (red/orange/blue banners)  
  - Loading spinner  

## Tech Stack

- **Language**: Python 3.9+  
- **ML Libraries**: scikit-learn, pandas, numpy  
- **Vectorization**: TfidfVectorizer  
- **Visualization**: matplotlib  
- **Web App**: Streamlit  
- **Environment**: Anaconda / Jupyter Notebook  

