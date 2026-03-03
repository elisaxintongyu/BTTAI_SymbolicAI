// app/api/ask/route.ts
import { NextRequest, NextResponse } from "next/server";

const PYTHON_BACKEND_URL =
  process.env.PYTHON_BACKEND_URL ?? "http://localhost:8000";

interface AskRequestBody {
  imageUrl: string;
  question: string;
}

export interface DetectedObject {
  label: string;
  bbox: [number, number, number, number]; // [x, y, w, h]
}

export interface AskResponseBody {
  objects: DetectedObject[];
  fol: string[];
  plan: string[];
  answer: string;
  detection_image_url?: string;
  grid_image_url?: string;
}

export async function POST(req: NextRequest): Promise<NextResponse> {
  try {
    const body = (await req.json()) as AskRequestBody;

    if (!body.imageUrl || !body.question) {
      return NextResponse.json(
        { error: "imageUrl and question are required" },
        { status: 400 }
      );
    }

    const pythonRes = await fetch(`${PYTHON_BACKEND_URL}/pipeline`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_url: body.imageUrl,
        question: body.question,
      }),
    });

    if (!pythonRes.ok) {
      const text = await pythonRes.text();
      console.error("Python backend error:", text);
      return NextResponse.json(
        { error: "Python backend error", details: text },
        { status: 500 }
      );
    }

    const data = (await pythonRes.json()) as AskResponseBody;

    return NextResponse.json<AskResponseBody>(data, { status: 200 });
  } catch (err) {
    console.error("Ask route error:", err);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
