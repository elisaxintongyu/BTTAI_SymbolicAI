// app/page.jsx
export default function Page() {
  return (
    <div className="min-h-screen bg-gray-100 flex">
      {/* LEFT PANEL */}
      <div className="w-[40%] max-w-xl border-r border-gray-300 flex flex-col bg-white">
        <ChatSection />
        <SceneList />
      </div>

      {/* RIGHT PANEL */}
      <div className="flex-1 p-8 overflow-y-auto bg-gray-50">
        <ModelPanel />
      </div>
    </div>
  );
}

/* ---------- LEFT: CHAT ---------- */

function ChatSection() {
  return (
    <div className="flex-1 p-6 flex flex-col justify-between">
      {/* Chat history */}
      <div>
        {/* Scene image placeholder */}
        <div className="mb-4">
          <div className="w-full h-40 rounded-md bg-gray-200 flex items-center justify-center text-gray-500 text-sm">
            Scene image
          </div>
        </div>

        {/* User message */}
        <div className="bg-[#5b4b7a] text-white px-4 py-3 rounded-2xl rounded-bl-sm w-fit mb-3 max-w-[90%]">
          Can the monkey get the bananas on the shelf?
        </div>

        {/* Model reply */}
        <div className="bg-gray-200 text-gray-900 px-4 py-3 rounded-2xl rounded-br-sm w-fit max-w-[90%]">
          It can if it stacks box A on B on C, and then climbs the stack.
        </div>
      </div>

      {/* Input area */}
      <div className="mt-4 flex items-center gap-3">
        {/* Plus button */}
        <button className="w-9 h-9 rounded-full bg-gray-200 flex items-center justify-center text-lg">
          +
        </button>

        {/* Input */}
        <div className="flex-1 flex items-center bg-[#efe8f5] rounded-full px-4 py-2 border border-gray-300">
          <input
            className="flex-1 bg-transparent border-none outline-none text-sm"
            placeholder="Message Neural Symbolic Monkeys"
          />
          {/* Search icon (simple text/emoji to avoid extra deps) */}
          <span className="text-lg mr-2">🔍</span>
          {/* Mic icon */}
          <span className="text-lg">🎤</span>
        </div>
      </div>
    </div>
  );
}

/* ---------- LEFT: SCENE LIST ---------- */

function SceneList() {
  const scenes = ["Scene…", "Scene…", "Scene", "Scene", "Scene"];

  return (
    <div className="border-t border-gray-200 px-4 py-3">
      {/* View toggle icons placeholder */}
      <div className="flex justify-end mb-3 gap-2 text-gray-500 text-sm">
        <div className="w-7 h-7 flex items-center justify-center rounded-md bg-gray-200">
          ≡
        </div>
        <div className="w-7 h-7 flex items-center justify-center rounded-md">
          ☐
        </div>
      </div>

      <div className="space-y-2">
        {scenes.map((label, i) => (
          <button
            key={i}
            className={`w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm ${
              i === 0 ? "bg-[#e6deef]" : "bg-transparent hover:bg-gray-100"
            }`}
          >
            <span className="w-3 h-3 rounded-full bg-gray-300 inline-block" />
            <span className="text-gray-800">{label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

/* ---------- RIGHT: MODEL PANEL ---------- */

function ModelPanel() {
  return (
    <div className="max-w-4xl mx-auto">
      <header className="mb-8">
        <h1 className="text-3xl font-semibold text-gray-900">
          Neural Symbolic Monkeys
        </h1>
        <p className="text-sm text-gray-500 mt-1">Model</p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-start">
        {/* Image block */}
        <div>
          <h2 className="text-sm font-semibold text-gray-600 mb-2">Image</h2>
          <div className="w-full aspect-[16/9] rounded-md bg-gray-200 flex items-center justify-center text-gray-500 text-sm">
            Scene image
          </div>
        </div>

        {/* Object detection grid */}
        <div>
          <h2 className="text-sm font-semibold text-gray-600 mb-2">
            Object Detection
          </h2>
          <ObjectGrid />
        </div>
      </div>

      {/* Directions block */}
      <div className="mt-10">
        <h2 className="text-sm font-semibold text-gray-700 mb-2">
          Directions
        </h2>
        <div className="bg-gray-100 rounded-md px-4 py-3 text-sm font-mono text-gray-800 space-y-1">
          <div>leftOf(monkey, boxA)</div>
          <div>get(monkey, banana)</div>
        </div>
      </div>
    </div>
  );
}

/* ---------- OBJECT GRID ---------- */

function ObjectGrid() {
  const rows = 10;
  const cols = 10;

  // crude “heatmap” layout similar to the screenshot
  const specialCells = {
    "1,7": "bg-amber-800",
    "4,6": "bg-amber-500",
    "4,5": "bg-yellow-400",
    "4,7": "bg-amber-700",
    "8,3": "bg-gray-400",
    "8,4": "bg-gray-500",
    "8,5": "bg-gray-600",
    "8,6": "bg-gray-700",
  };

  return (
    <div className="inline-grid grid-cols-10 gap-[1px] bg-black/40 p-[2px]">
      {Array.from({ length: rows }).map((_, row) =>
        Array.from({ length: cols }).map((_, col) => {
          const key = `${col},${row}`;
          const color = specialCells[key] || "bg-black/10";

          return (
            <div
              key={key}
              className={`w-5 h-5 md:w-6 md:h-6 border border-black/40 ${color}`}
            />
          );
        })
      )}
    </div>
  );
}
