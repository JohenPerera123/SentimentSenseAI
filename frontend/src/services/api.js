const API_BASE = 'http://localhost:8000/api';

export const fetchModels = async () => {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error('Failed to fetch models');
  return res.json();
};

export const fetchMetrics = async () => {
  const res = await fetch(`${API_BASE}/metrics`);
  if (!res.ok) throw new Error('Failed to fetch metrics');
  return res.json();
};

export const predictSentiment = async (text, model, domain) => {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, model, domain })
  });
  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || 'Failed to predict sentiment');
  }
  return res.json();
};

export const compareModels = async (text, domain) => {
  const res = await fetch(`${API_BASE}/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, domain })
  });
  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || 'Failed to compare models');
  }
  return res.json();
};
