import React, { useState } from 'react';
import { compareModels } from '../services/api';
import { Loader2, GitCompare, AlertCircle } from 'lucide-react';

const Compare = () => {
  const [text, setText] = useState('');
  const [domain, setDomain] = useState('imdb');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);

  const handleCompare = async () => {
    if (!text.trim()) {
      setError('Please enter some text to compare.');
      return;
    }
    
    setLoading(true);
    setError(null);
    setResults(null);
    
    try {
      const data = await compareModels(text, domain);
      setResults(data.predictions);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getAgreementStatus = (preds) => {
    const sentiments = preds.map(p => p.sentiment);
    const unique = new Set(sentiments);
    if (unique.size === 1) {
      return { msg: `Model Agreement: ${preds.length} / ${preds.length}`, color: '#10B981', agree: true };
    }
    return { msg: "Model Disagreement Detected", color: '#F59E0B', agree: false };
  };

  return (
    <div className="container">
      <h1 className="page-title">Compare Models</h1>
      <p className="page-subtitle">Submit one text and see how Traditional ML and Transformers differ.</p>

      <div className="glass-panel" style={{ padding: '2rem', marginBottom: '2rem' }}>
        <div className="form-group">
          <label className="form-label">Input Text</label>
          <textarea 
            className="text-area"
            placeholder="The movie was surprisingly good!"
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
              </div>
            </div>
            <div className={`radio-card ${domain === 'twitter' ? 'selected' : ''}`} onClick={() => setDomain('twitter')}>
              <input type="radio" checked={domain === 'twitter'} readOnly style={{ display: 'none' }} />
              <div>
                <strong>Twitter</strong>
              </div>
            </div>
          </div>
        </div>

        {error && (
          <div className="error-message">
            <AlertCircle size={18} style={{ verticalAlign: 'middle', marginRight: '8px' }} />
            {error}
          </div>
        )}

        <button className="btn" onClick={handleCompare} disabled={loading}>
          {loading ? <Loader2 className="spinner" size={20} /> : <GitCompare size={20} />}
          {loading ? 'Comparing Models...' : 'Run Comparison'}
        </button>
      </div>

      {results && results.length > 0 && (
        <div className="glass-panel prediction-result" style={{ padding: '2rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
            <h2>Comparison Results</h2>
            {(() => {
              const status = getAgreementStatus(results);
              return (
                <div style={{ 
                  background: `${status.color}22`, 
                  color: status.color, 
                  padding: '0.5rem 1rem', 
                  borderRadius: '8px',
                  fontWeight: 'bold' 
                }}>
                  {status.msg}
                </div>
              );
            })()}
          </div>
          
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Prediction</th>
                  <th>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r, i) => (
                  <tr key={i}>
                    <td><strong>{r.model.replace('_', ' ').toUpperCase()}</strong></td>
                    <td>
                      <span className={`sentiment-badge ${r.sentiment === 'positive' ? 'positive' : 'negative'}`} style={{ padding: '0.25rem 0.5rem', fontSize: '0.9rem' }}>
                        {r.sentiment.toUpperCase()}
                      </span>
                    </td>
                    <td>{(r.confidence * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default Compare;
