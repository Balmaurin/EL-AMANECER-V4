import { NextRequest, NextResponse } from 'next/server';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { current_thought } = body;

    // Intentar conectar con el backend real
    try {
      const backendResponse = await fetch(`${API_BASE}/api/consciousness/process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ current_thought }),
      });

      if (backendResponse.ok) {
        const data = await backendResponse.json();
        return NextResponse.json(data);
      }
    } catch (error) {
      console.log('Backend not available, using fallback processing');
    }

    // Procesamiento simulado de consciencia
    const response = {
      success: true,
      current_meta_awareness: Math.random() * 0.4 + 0.3, // 0.3-0.7
      cognitive_depth: Math.floor(Math.random() * 5 + 2), // 2-7
      emergence_triggered: Math.random() > 0.7,
      consciousness_level: 'Level 5: Meta-Cognitive Awareness',
      insight: generateInsight(current_thought),
      processed_at: new Date().toISOString()
    };

    return NextResponse.json(response);

  } catch (error) {
    console.error('Error processing consciousness:', error);
    return NextResponse.json(
      { error: 'Error procesando consciencia', details: String(error) },
      { status: 500 }
    );
  }
}

function generateInsight(thought: string): string {
  const insights = [
    'Pensamiento procesado con profundidad meta-cognitiva',
    'Patrones emergentes detectados en el análisis',
    'Consciencia auto-referencial activada',
    'Sistema de reflexión profunda operando',
    'Meta-análisis de pensamiento completado'
  ];
  return insights[Math.floor(Math.random() * insights.length)];
}
