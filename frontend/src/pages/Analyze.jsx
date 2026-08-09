import React, { useState } from 'react';
import { predictSentiment } from '../services/api';
import { Loader2, MessageSquare, AlertCircle } from 'lucide-react';

const Analyze = () => {
  const [text, setText] = useState('');
  const [domain, setDomain] = useState('imdb');
  const [model, setModel] = useState('logistic_regression');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleAnalyze = async () => {
    if (!text.trim()) {
      setError('Please enter some text to analyze.');
      return;
    }
    
    setLoading(true);
    setError(null);
    setResult(null);
    
    try {
      const data = await predictSentiment(text, model, domain);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1 className="page-title">Sentiment Analysis</h1>
      <p className="page-subtitle">Test out our NLP models on movie reviews or social media text.</p>

      <div className="glass-panel" style={{ padding: '2rem', marginBottom: '2rem' }}>
        <div className="form-group">
          <label className="form-label">Input Text</label>
          <textarea 
            className="text-area"
            placeholder="Enter a movie review or social media text..."
            value={text}
            onChange={(e) => setText(e.target.value)}
            maxLength={5000}
          />
        </div>

        <div className="form-group">
          <label className="form-label">Domain</label>
          <div className="select-group">
            <div className={`radio-card ${domain === 'imdb' ? 'selected' : ''}`} onClick={() => setDomain('imdb')}>
              <input type="radio" checked={domain === 'imdb'} readOnly style={{ display: 'none' }} />
              <div>
                <strong>IMDb</strong>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Movie Reviews</div>
              </div>
            </div>
            <div className={`radio-card ${domain === 'twitter' ? 'selected' : ''}`} onClick={() => setDomain('twitter')}>
              <input type="radio" checked={domain === 'twitter'} readOnly style={{ display: 'none' }} />
              <div>
                <strong>Twitter</strong>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Social Media</div>
              </div>
            </div>
          </div>
        </div>

        <div className="form-group">
          <label className="form-label">Model</label>
          <div className="select-group">
            <div className={`radio-card ${model === 'logistic_regression' ? 'selected' : ''}`} onClick={() => setModel('logistic_regression')}>
              <strong>Logistic Regression</strong>
            </div>
            <div className={`radio-card ${model === 'linear_svm' ? 'selected' : ''}`} onClick={() => setModel('linear_svm')}>
              <strong>Linear SVM</strong>
            </div>
            <div className={`radio-card ${model === 'distilbert' ? 'selected' : ''}`} onClick={() => setModel('distilbert')}>
              <strong>DistilBERT</strong>
            </div>
          </div>
        </div>

        {error && (
          <div className="error-message">
            <AlertCircle size={18} style={{ verticalAlign: 'middle', marginRight: '8px' }} />
            {error}
          </div>
        )}

        <button className="btn" onClick={handleAnalyze} disabled={loading}>
          {loading ? <Loader2 className="spinner" size={20} /> : <MessageSquare size={20} />}
          {loading ? 'Analyzing...' : 'Analyze Sentiment'}
        </button>
      </div>

      {result && (
        <div className="glass-panel prediction-result" style={{ padding: '2rem' }}>
          <h2 style={{ marginBottom: '1.5rem' }}>Analysis Results</h2>
          <div className="dashboard-grid">
            <div className="stat-card glass-panel" style={{ background: 'rgba(15, 23, 42, 0.4)' }}>
              <div className="stat-title">Sentiment</div>
              <div className={`sentiment-badge ${result.sentiment === 'positive' ? 'positive' : 'negative'}`}>
                {result.sentiment.toUpperCase()}
              </div>
            </div>
            <div className="stat-card glass-panel" style={{ background: 'rgba(15, 23, 42, 0.4)' }}>
              <div className="stat-title">Confidence</div>
              <div className="stat-value">{(result.confidence * 100).toFixed(1)}%</div>
            </div>
            <div className="stat-card glass-panel" style={{ background: 'rgba(15, 23, 42, 0.4)' }}>
              <div className="stat-title">Configuration</div>
              <div style={{ color: 'var(--text-muted)' }}>
                Model: <strong>{result.model}</strong><br/>
                Domain: <strong>{result.domain}</strong>
              </div>
            </div>
          </div>
          <div style={{ marginTop: '2rem' }}>
            <h4 style={{ color: 'var(--text-muted)', textTransform: 'uppercase', fontSize: '0.8rem', marginBottom: '0.5rem' }}>Original Text</h4>
            <p style={{ fontStyle: 'italic', background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '8px' }}>"{result.text}"</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default Analyze;
