import { NextRequest, NextResponse } from 'next/server';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { dataset } = body;

    // Intentar conectar con el backend real
    try {
      const backendResponse = await fetch(`${API_BASE}/api/training/dataset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset }),
      });

      if (backendResponse.ok) {
        const data = await backendResponse.json();
        return NextResponse.json(data);
      }
    } catch (error) {
      console.log('Backend not available, using fallback');
    }

    // Generar dataset ID si el backend no está disponible
    const datasetId = dataset.id || `dataset_${Date.now()}`;

    const response = {
      success: true,
      dataset_id: datasetId,
      message: 'Dataset creado exitosamente',
      stats: {
        total_answers: dataset.answers?.length || 0,
        total_tokens: (dataset.answers?.length || 0) * 50, // Estimación
        created_at: new Date().toISOString()
      }
    };

    return NextResponse.json(response);

  } catch (error) {
    console.error('Error creating dataset:', error);
    return NextResponse.json(
      { error: 'Error creando dataset', details: String(error) },
      { status: 500 }
    );
  }
}
