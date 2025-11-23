import { NextResponse } from "next/server";

const API_BASE = process.env.NEXT_PUBLIC_SHEILY_API || "http://localhost:8001";

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const r = await fetch(`${API_BASE}/api/chat/`, { // barra final para evitar 307
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const t = await r.text();
      return NextResponse.json({ error: t }, { status: r.status });
    }
    const data = await r.json();
    // normaliza {reply|text|answer|message.content}
    const reply = data.reply ?? data.text ?? data.answer ?? data?.message?.content ?? "";
    return NextResponse.json({ reply, raw: data });
  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: 500 });
  }
}
