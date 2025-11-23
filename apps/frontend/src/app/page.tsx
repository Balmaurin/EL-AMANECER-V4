'use client';

import { useEffect } from 'react';

export default function Dashboard() {
  useEffect(() => {
    // Redirigir inmediatamente a sheily-web.html
    if (typeof window !== 'undefined') {
      window.location.replace('/sheily-web.html');
    }
  }, []);

  return (
    <div style={{
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      height: '100vh',
      background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
      color: '#f1f5f9',
      fontFamily: 'system-ui, -apple-system, sans-serif'
    }}>
      <div style={{ textAlign: 'center' }}>
        <div style={{
          fontSize: '3rem',
          marginBottom: '20px',
          animation: 'pulse 2s ease-in-out infinite'
        }}>
          🧠
        </div>
        <h1 style={{ fontSize: '2rem', marginBottom: '10px' }}>
          Cargando Sheily AI Dashboard
        </h1>
        <p style={{ color: '#94a3b8', fontSize: '1.1rem' }}>
          Redirigiendo al sistema MCP-Phoenix...
        </p>
      </div>
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.7; transform: scale(1.1); }
        }
      `}</style>
    </div>
  );
}
