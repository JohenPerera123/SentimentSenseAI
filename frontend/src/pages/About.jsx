import React from 'react';

const About = () => {
  return (
    <div className="container">
      <h1 className="page-title">About SentimentScope</h1>
      <p className="page-subtitle">Multi-Model NLP Sentiment Analysis Project</p>

      <div className="glass-panel" style={{ padding: '2rem', marginBottom: '2rem' }}>
        <h2 style={{ marginBottom: '1rem', color: 'var(--primary)' }}>1. Problem Statement</h2>
        <p style={{ lineHeight: '1.6', color: 'var(--text-muted)' }}>
          Sentiment analysis in Natural Language Processing (NLP) often faces challenges when switching domains. 
          Models that perform exceptionally well on long-form, highly structured text (like movie reviews) may struggle with short-form, informal text containing slang and complex context (like social media). 
          This project aims to explore the efficacy of traditional Machine Learning against modern Transformers across these domains.
        </p>
      </div>

      <div className="glass-panel" style={{ padding: '2rem', marginBottom: '2rem' }}>
        <h2 style={{ marginBottom: '1rem', color: 'var(--primary)' }}>2. Datasets</h2>
        <ul style={{ lineHeight: '1.8', color: 'var(--text-muted)', marginLeft: '1.5rem' }}>
          <li><strong>IMDb Movie Reviews:</strong> A highly structured dataset of 50,000 long-form movie reviews labeled as positive or negative.</li>
          <li><strong>Twitter Sentiment140:</strong> Originating from approximately 1.6 million original records, this project uses a 50,000-record balanced sample of short-form, informal tweets labeled as positive or negative.</li>
        </ul>
      </div>

      <div className="dashboard-grid">
        <div className="glass-panel" style={{ padding: '2rem' }}>
          <h2 style={{ marginBottom: '1rem', color: 'var(--primary)' }}>3. Traditional Pipeline</h2>
          <p style={{ lineHeight: '1.6', color: 'var(--text-muted)' }}>
            <strong>NLP Preprocessing & TF-IDF:</strong> Text was rigorously cleaned (removing HTML tags, URLs, non-alphabetic characters) and tokenized. 
            Words were lemmatized using NLTK to reduce dimensionality. 
            The cleaned text was then converted to numerical vectors using TF-IDF (Term Frequency - Inverse Document Frequency).<br/><br/>
            <strong>Traditional ML:</strong> We trained Naive Bayes, Logistic Regression, and Linear SVM models on approximately 40,000 samples. 
            These models are incredibly fast and establish a powerful baseline.
          </p>
        </div>

        <div className="glass-panel" style={{ padding: '2rem' }}>
          <h2 style={{ marginBottom: '1rem', color: 'var(--secondary)' }}>4. Transformer Pipeline</h2>
          <p style={{ lineHeight: '1.6', color: 'var(--text-muted)' }}>
            <strong>DistilBERT:</strong> We integrated the `distilbert-base-uncased` transformer from Hugging Face. 
            Unlike TF-IDF, which relies on word frequencies, Transformers process text contextually (understanding negation and sequential phrasing).<br/><br/>
            <strong>Hardware Limitation & Fair Evaluation:</strong> Fine-tuning Transformers is computationally expensive. Due to CPU constraints, we trained a "Fair" subset of 5,000 samples for DistilBERT. 
            Despite seeing only a fraction of the data compared to the traditional models, the Transformer showed incredible capabilities.
          </p>
        </div>
      </div>

      <div className="glass-panel" style={{ padding: '2rem', marginTop: '2rem' }}>
        <h2 style={{ marginBottom: '1rem', color: 'var(--primary)' }}>Key Findings</h2>
        <ul style={{ lineHeight: '1.8', color: 'var(--text-muted)', marginLeft: '1.5rem' }}>
          <li><strong>Traditional ML is still viable:</strong> On the IMDb dataset, Logistic Regression trained on 40,000 samples achieved a stunning ~90.7% F1-score, proving TF-IDF remains highly competitive for structured, long-form text.</li>
          <li><strong>Transformers rule informal context:</strong> On the Twitter dataset, DistilBERT (trained on only 5,000 samples) achieved ~80.3% F1, completely outperforming Logistic Regression (77.3% trained on 40,000 samples). Contextual embeddings are vital for slang and informal phrasing.</li>
          <li><strong>Hardware scale matters:</strong> The primary limiting factor for Transformers remains compute cost. The DistilBERT models deployed here are completely functional and ready to scale with GPU resources.</li>
        </ul>
      </div>
    </div>
  );
};

export default About;
