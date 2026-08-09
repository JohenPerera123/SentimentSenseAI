import React, { useEffect, useState } from 'react';
import { fetchMetrics } from '../services/api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, CartesianAxis } from 'recharts';

const Performance = () => {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchMetrics().then(data => {
      setMetrics(data);
      setLoading(false);
    }).catch(err => {
      console.error(err);
      setLoading(false);
    });
  }, []);

  if (loading) {
    return <div className="container" style={{ textAlign: 'center', marginTop: '4rem' }}>Loading metrics...</div>;
  }

  if (!metrics) return null;

  const prepareChartData = (domainData) => {
    return [
      { name: 'Naive Bayes', f1: domainData.naive_bayes.f1 * 100 },
      { name: 'Logistic Regression', f1: domainData.logistic_regression.f1 * 100 },
      { name: 'Linear SVM', f1: domainData.linear_svm.f1 * 100 },
      { name: 'DistilBERT (Fair 5k)', f1: domainData.distilbert.f1 * 100 }
    ];
  };

  const imdbData = prepareChartData(metrics.imdb);
  const twitterData = prepareChartData(metrics.twitter);

  const scalingData = [
    { name: 'IMDb', 'Pilot (500)': metrics.imdb.distilbert_pilot.f1 * 100, 'Fair 5k (5,000)': metrics.imdb.distilbert.f1 * 100 },
    { name: 'Twitter', 'Pilot (500)': metrics.twitter.distilbert_pilot.f1 * 100, 'Fair 5k (5,000)': metrics.twitter.distilbert.f1 * 100 }
  ];

  return (
    <div className="container">
      <h1 className="page-title">Performance Dashboard</h1>
      <p className="page-subtitle">Verified test-set results from Phase 3.5 and Phase 4.5</p>

      <div className="dashboard-grid">
        <div className="stat-card glass-panel">
          <div className="stat-title">IMDb Best Model</div>
          <div className="stat-value">Logistic Regression</div>
          <div style={{ color: 'var(--secondary)' }}>F1: 90.72%</div>
        </div>
        <div className="stat-card glass-panel">
          <div className="stat-title">Twitter Best Model</div>
          <div className="stat-value">DistilBERT</div>
          <div style={{ color: 'var(--secondary)' }}>F1: 80.34%</div>
        </div>
      </div>

      <div className="glass-panel" style={{ padding: '1.5rem', marginBottom: '2rem', background: 'rgba(79, 70, 229, 0.1)', border: '1px solid rgba(79, 70, 229, 0.3)' }}>
        <h4 style={{ color: '#818cf8', marginBottom: '0.5rem' }}>Experimental Context</h4>
        <p style={{ fontSize: '0.9rem', color: 'var(--text-main)', lineHeight: '1.5' }}>
          Traditional models were trained using approximately 40,000 training samples, while the controlled DistilBERT experiment used 5,000 training samples. Results should therefore be interpreted with this difference in training scale in mind.
        </p>
      </div>

      <div className="dashboard-grid">
        <div className="glass-panel" style={{ padding: '2rem' }}>
          <h3 style={{ marginBottom: '1.5rem' }}>Model F1-Score: IMDb</h3>
          <div style={{ height: 300 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={imdbData} margin={{ top: 5, right: 30, left: -20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="name" stroke="#94a3b8" tick={{ fontSize: 12 }} />
                <YAxis domain={[75, 100]} stroke="#94a3b8" />
                <Tooltip contentStyle={{ background: '#1e293b', border: 'none', borderRadius: '8px' }} />
                <Bar dataKey="f1" fill="#4F46E5" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '2rem' }}>
          <h3 style={{ marginBottom: '1.5rem' }}>Model F1-Score: Twitter</h3>
          <div style={{ height: 300 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={twitterData} margin={{ top: 5, right: 30, left: -20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="name" stroke="#94a3b8" tick={{ fontSize: 12 }} />
                <YAxis domain={[60, 85]} stroke="#94a3b8" />
                <Tooltip contentStyle={{ background: '#1e293b', border: 'none', borderRadius: '8px' }} />
                <Bar dataKey="f1" fill="#10B981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="glass-panel" style={{ padding: '2rem', marginTop: '2rem' }}>
        <h3 style={{ marginBottom: '0.5rem' }}>DistilBERT Training-Data Scaling Experiment</h3>
        <p style={{ color: 'var(--text-muted)', marginBottom: '1.5rem', fontSize: '0.9rem' }}>Comparing the 500-sample pilot against the 5,000-sample fair evaluation.</p>
        <div style={{ height: 350 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={scalingData} margin={{ top: 5, right: 30, left: -20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="name" stroke="#94a3b8" />
              <YAxis domain={[60, 90]} stroke="#94a3b8" />
              <Tooltip contentStyle={{ background: '#1e293b', border: 'none', borderRadius: '8px' }} />
              <Legend wrapperStyle={{ paddingTop: '20px' }} />
              <Bar dataKey="Pilot (500)" fill="#64748b" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Fair 5k (5,000)" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default Performance;
