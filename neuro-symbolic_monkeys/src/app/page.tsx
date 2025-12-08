"use client";

import { useState, ChangeEvent } from "react";
import bgImage from "./Screenshot 2025-12-07 at 10.01.10 PM.png";
import detectedImg from "./vis_yolo.jpg"

/* -------------------------------------------------------
   TYPES
-------------------------------------------------------- */

interface ChatSectionProps {
  sceneImage: string | null;
  setSceneImage: (url: string | null) => void;
}

interface ModelPanelProps {
  sceneImage: string | null;
}

/* -------------------------------------------------------
   ROOT PAGE
-------------------------------------------------------- */

export default function Page(): JSX.Element {
  const [sceneImage, setSceneImage] = useState<string | null>(null);

  return (
    <div className="min-h-screen bg-gray-100 flex">
      {/* LEFT PANEL */}
      <div className="w-[40%] max-w-xl border-r border-gray-300 flex flex-col bg-white">
        <ChatSection sceneImage={sceneImage} setSceneImage={setSceneImage} />
      </div>

      {/* RIGHT PANEL */}
      <div className="flex-1 p-8 overflow-y-auto bg-gray-50">
        <ModelPanel sceneImage={sceneImage} />
      </div>
    </div>
  );
}

/* -------------------------------------------------------
   CHAT + IMAGE UPLOAD
-------------------------------------------------------- */

function ChatSection({
  sceneImage,
  setSceneImage,
}: ChatSectionProps): JSX.Element {
  const [text, setText] = useState<string>("");
  const uploadImage = async (file: File): Promise<void> => {
    const formData = new FormData();
    formData.append("image", file);

    const res = await fetch("/api/upload", {
      method: "POST",
      body: formData,
    });

    const data = (await res.json()) as { fileUrl?: string; error?: string };

    if (res.ok && data.fileUrl) {
      setSceneImage(data.fileUrl);
    } else {
      alert("Upload failed: " + (data.error ?? "Unknown error"));
    }
  };

  const handleUploadClick = (): void => {
    const input = document.getElementById("imageUploadInput") as HTMLInputElement | null;
    input?.click();
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>): void => {
    const file = e.target.files?.[0];
    if (file) uploadImage(file);
  };

  const handleMessaging = async (): Promise<void> => {
    if (!sceneImage) {
      alert("Please upload an image first.");
      return;
    }
    const question = text.trim();
    if (!question) {
      alert("Please enter a question.");
      return;
    }

    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ imageUrl: sceneImage, question }),
    });
    const data = await res.json();
    //TODO: handle response
    // data.objects, data.fol, data.plan, data.answer

  };

  return (
    <div className="flex-1 p-6 flex flex-col justify-between">
      {/* Image block */}
        <div>
          <h2 className="text-sm font-semibold text-gray-600 mb-2">Image</h2>

          {sceneImage ? (
            <img
              src={sceneImage}
              className="w-full rounded-md object-cover h-[50vh]"
              alt="Uploaded Scene"
            />
          ) : (
            <div className="w-full min-h-[40vh] rounded-md bg-gray-200 flex items-center justify-center text-gray-500 text-sm">
              Scene image
            </div>
          )}
        </div>
      {/* Chat history */}
      <div>
        <div className="bg-[#5b4b7a] text-white px-4 py-3 rounded-2xl rounded-bl-sm w-fit mb-3 max-w-[90%]">
          Welcome to Neural Symbolic Monkeys! Please start with uploading an image of a scene.
        </div>

        <div className="bg-gray-200 text-gray-900 px-4 py-3 rounded-2xl rounded-br-sm w-fit mb-3 max-w-[90%] justify-self-end">
          How does the monkey get the banana?
        </div>

        <div className="bg-[#5b4b7a] text-white px-4 py-3 rounded-2xl rounded-bl-sm w-fit mb-3 max-w-[90%]">
          It can climb on top of box B and move right to reach the banana.
        </div>
      </div>

      {/* Input area */}
      <div className="mt-4 flex items-center gap-3">
        <input
          id="imageUploadInput"
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleFileChange}
        />

        {/* Plus upload button */}
        <button
          onClick={handleUploadClick}
          className="w-full py-3 rounded-lg text-black bg-gray-200 flex items-center justify-center text-lg font-medium hover:bg-gray-300 transition"
        >
          + Upload Image
        </button>

        <div className="flex-1 flex items-center text-black bg-[#efe8f5] rounded-full px-4 py-2 border border-gray-300">
          <input
            className="flex-1 bg-transparent border-none outline-none text-sm"
            placeholder="Message Neural Symbolic Monkeys..."
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)} 
          />
          <button
            onClick={handleMessaging}
            className="ml-2 px-4 py-2 bg-blue-500 text-white rounded-full hover:bg-blue-600 transition"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

/* -------------------------------------------------------
   RIGHT: MODEL PANEL
-------------------------------------------------------- */

function ModelPanel({ sceneImage }: ModelPanelProps): JSX.Element {
  return (
    <div className="max-w-4xl mx-auto">
      <header className="mb-8">
        <h1 className="text-3xl font-semibold text-gray-900">
          Neural Symbolic Monkeys
        </h1>
        <p className="text-sm text-gray-500 mt-1">Model</p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-start">
        <div>
          <h2 className="text-sm font-semibold text-gray-600 mb-2">
            Detection with bounded boxes 
          </h2>
          <img src={detectedImg.src} alt="Detected Scene" />
        </div>

        {/* Object detection grid */}
        <div>
          <h2 className="text-sm font-semibold text-gray-600 mb-2">
            Detection with array representation
          </h2>
          <img src={bgImage.src} alt="Object Detection Grid" />
        </div>
      </div>

      <div className="mt-10">
        <h2 className="text-sm font-semibold text-gray-700 mb-2">FOL Directions</h2>
        <div className="bg-gray-100 rounded-md px-4 py-3 text-sm font-mono text-gray-800 space-y-1">
          <div>(move monkey l1 l2)</div>
          <div>(climb_on monkey box1 l2)</div>
          <div>(climb_off monkey box1 l2)</div>
          <div>(move monkey l2 l3)</div>
          <div>(grab_banana_from_ground monkey banana l3)</div>
        </div>
      </div>
    </div>
  );
}

/* -------------------------------------------------------
   OBJECT GRID
-------------------------------------------------------- */

function ObjectGrid(): JSX.Element {
  const rows = 10;
  const cols = 10;

  const specialCells: Record<string, string> = {
    "1,7": "bg-amber-800",
    "4,6": "bg-amber-500",
    "4,5": "bg-yellow-400",
    "4,7": "bg-amber-700",
    "8,3": "bg-gray-400",
    "8,4": "bg-gray-500",
    "8,5": "bg-gray-600",
    "8,6": "bg-gray-700",
  };

  const grid = [];

  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const key = `${col},${row}`;
      const color = specialCells[key] ?? "bg-black/10";

      grid.push(
        <div
          key={key}
          className={`w-5 h-5 md:w-6 md:h-6 border border-black/40 ${color}`}
        />
      );
    }
  }

  return (
    <div className="inline-grid grid-cols-10 gap-[1px] bg-black/40 p-[2px]">
      {grid}
    </div>
  );
}
