'use client';

import { useEffect } from 'react';

export default function Home() {
  useEffect(() => {
    // Redirect to Chat_Editado_1.html
    window.location.href = '/Chat_Editado_1.html';
  }, []);

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      height: '100vh',
      fontFamily: 'Arial, sans-serif'
    }}>
      <p>Redirigiendo a SHEILY Chat...</p>
    </div>
  );
}
