"use client";

import { useState, ChangeEvent } from "react";
import bgImage from "./Screenshot 2025-12-07 at 10.01.10 PM.png";
import detectedImg from "./vis_yolo.jpg";

interface DetectedObject {
  label: string;
  bbox: [number, number, number, number];
}

interface AskResponseBody {
  objects: DetectedObject[];
  fol: string[];
  plan: string[];
  answer: string;
}

interface Message {
  role: "assistant" | "user";
  text: string;
}

interface ChatSectionProps {
  sceneImage: string | null;
  setSceneImage: (url: string | null) => void;
  messages: Message[];
  setMessages: (messages: Message[]) => void;
  setResult: (result: AskResponseBody | null) => void;
}

interface ModelPanelProps {
  result: AskResponseBody | null;
}

export default function Page(): JSX.Element {
  const [sceneImage, setSceneImage] = useState<string | null>(null);
  const [result, setResult] = useState<AskResponseBody | null>(null);
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      text: "Welcome to Neural Symbolic Monkeys! Please upload an image and ask a question.",
    },
  ]);

  return (
    <div className="min-h-screen bg-gray-100 flex">
      <div className="w-[40%] max-w-xl border-r border-gray-300 flex flex-col bg-white">
        <ChatSection
          sceneImage={sceneImage}
          setSceneImage={setSceneImage}
          messages={messages}
          setMessages={setMessages}
          setResult={setResult}
        />
      </div>

      <div className="flex-1 p-8 overflow-y-auto bg-gray-50">
        <ModelPanel result={result} />
      </div>
    </div>
  );
}

function ChatSection({
  sceneImage,
  setSceneImage,
  messages,
  setMessages,
  setResult,
}: ChatSectionProps): JSX.Element {
  const [text, setText] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);

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
      setResult(null);
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
    if (file) {
      void uploadImage(file);
    }
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

    const nextMessages: Message[] = [...messages, { role: "user", text: question }];
    setMessages(nextMessages);
    setText("");
    setIsLoading(true);

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ imageUrl: sceneImage, question }),
      });

      const data = (await res.json()) as AskResponseBody & { error?: string; details?: string };
      if (!res.ok) {
        const msg = data.error ?? "Request failed";
        setMessages([
          ...nextMessages,
          { role: "assistant", text: `Error: ${msg}` },
        ]);
        return;
      }

      setResult(data);
      setMessages([
        ...nextMessages,
        { role: "assistant", text: data.answer || "No answer generated." },
      ]);
    } catch {
      setMessages([
        ...nextMessages,
        { role: "assistant", text: "Error: Could not reach backend." },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex-1 p-6 flex flex-col justify-between gap-4">
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

      <div className="overflow-y-auto max-h-[30vh] pr-1">
        {messages.map((message, index) => {
          const isAssistant = message.role === "assistant";
          return (
            <div
              key={`${message.role}-${index}`}
              className={`${
                isAssistant
                  ? "bg-[#5b4b7a] text-white rounded-bl-sm"
                  : "bg-gray-200 text-gray-900 rounded-br-sm ml-auto"
              } px-4 py-3 rounded-2xl w-fit mb-3 max-w-[90%]`}
            >
              {message.text}
            </div>
          );
        })}
        {isLoading && (
          <div className="bg-[#5b4b7a] text-white px-4 py-3 rounded-2xl rounded-bl-sm w-fit mb-3 max-w-[90%]">
            Thinking...
          </div>
        )}
      </div>

      <div className="mt-2 flex items-center gap-3">
        <input
          id="imageUploadInput"
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleFileChange}
        />

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
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                void handleMessaging();
              }
            }}
          />
          <button
            onClick={() => {
              void handleMessaging();
            }}
            disabled={isLoading}
            className="ml-2 px-4 py-2 bg-blue-500 text-white rounded-full hover:bg-blue-600 transition disabled:bg-blue-300"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

function ModelPanel({ result }: ModelPanelProps): JSX.Element {
  return (
    <div className="max-w-4xl mx-auto">
      <header className="mb-8">
        <h1 className="text-3xl font-semibold text-gray-900">Neural Symbolic Monkeys</h1>
        <p className="text-sm text-gray-500 mt-1">Model</p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-start">
        <div>
          <h2 className="text-sm font-semibold text-gray-600 mb-2">Detection with bounded boxes</h2>
          <img src={detectedImg.src} alt="Detected Scene" />
        </div>

        <div>
          <h2 className="text-sm font-semibold text-gray-600 mb-2">Detection with array representation</h2>
          <img src={bgImage.src} alt="Object Detection Grid" />
        </div>
      </div>

      <div className="mt-10 grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h2 className="text-sm font-semibold text-gray-700 mb-2">Detected Objects</h2>
          <div className="bg-gray-100 rounded-md px-4 py-3 text-sm font-mono text-gray-800 space-y-1">
            {(result?.objects ?? []).length === 0 ? (
              <div>No detections yet.</div>
            ) : (
              result?.objects.map((obj, idx) => (
                <div key={`${obj.label}-${idx}`}>
                  {obj.label}: [{obj.bbox.map((v) => v.toFixed(1)).join(", ")}]
                </div>
              ))
            )}
          </div>
        </div>

        <div>
          <h2 className="text-sm font-semibold text-gray-700 mb-2">FOL Predicates</h2>
          <div className="bg-gray-100 rounded-md px-4 py-3 text-sm font-mono text-gray-800 space-y-1">
            {(result?.fol ?? []).length === 0 ? (
              <div>No FOL yet.</div>
            ) : (
              result?.fol.map((line, idx) => <div key={`${line}-${idx}`}>{line}</div>)
            )}
          </div>
        </div>

        <div>
          <h2 className="text-sm font-semibold text-gray-700 mb-2">Plan</h2>
          <div className="bg-gray-100 rounded-md px-4 py-3 text-sm font-mono text-gray-800 space-y-1">
            {(result?.plan ?? []).length === 0 ? (
              <div>No plan yet.</div>
            ) : (
              result?.plan.map((line, idx) => <div key={`${line}-${idx}`}>{line}</div>)
            )}
          </div>
        </div>

        <div>
          <h2 className="text-sm font-semibold text-gray-700 mb-2">Answer</h2>
          <div className="bg-gray-100 rounded-md px-4 py-3 text-sm text-gray-800 min-h-[3rem]">
            {result?.answer ?? "No answer yet."}
          </div>
        </div>
      </div>
    </div>
  );
}
