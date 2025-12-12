import { createFileRoute } from "@tanstack/react-router"
import { useState, useMemo } from "react"

export const Route = createFileRoute("/learn/3-multihead")({
  component: MultiHeadLesson,
})

function MultiHeadLesson() {
  const [activeHead, setActiveHead] = useState<number | null>(null)
  const words = ["The", "bank", "of", "the", "river"]
  const numHeads = 4

  // Simulated attention patterns for different heads
  const headPatterns = useMemo(() => {
    const patterns = [
      // Head 1: Semantic (bank → river)
      [
        [0.8, 0.05, 0.05, 0.05, 0.05],
        [0.05, 0.1, 0.05, 0.05, 0.75],
        [0.1, 0.1, 0.6, 0.1, 0.1],
        [0.8, 0.05, 0.05, 0.05, 0.05],
        [0.05, 0.7, 0.05, 0.05, 0.15],
      ],
      // Head 2: Syntactic (determiner → noun)
      [
        [0.1, 0.7, 0.1, 0.05, 0.05],
        [0.3, 0.4, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.1, 0.6],
        [0.1, 0.1, 0.1, 0.1, 0.6],
        [0.1, 0.1, 0.6, 0.1, 0.1],
      ],
      // Head 3: Position (previous word)
      [
        [0.8, 0.1, 0.05, 0.03, 0.02],
        [0.7, 0.2, 0.05, 0.03, 0.02],
        [0.1, 0.7, 0.1, 0.05, 0.05],
        [0.05, 0.1, 0.7, 0.1, 0.05],
        [0.05, 0.05, 0.1, 0.7, 0.1],
      ],
      // Head 4: Long-range
      [
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.1, 0.1, 0.1, 0.1, 0.6],
        [0.6, 0.1, 0.1, 0.1, 0.1],
        [0.1, 0.6, 0.1, 0.1, 0.1],
        [0.6, 0.1, 0.1, 0.1, 0.1],
      ],
    ]
    return patterns
  }, [])

  const headDescriptions = [
    { name: "Semantic", desc: "Meaning relationships", color: "blue", example: "bank → river" },
    { name: "Syntactic", desc: "Grammar patterns", color: "purple", example: "the → noun" },
    { name: "Positional", desc: "Previous word focus", color: "orange", example: "word[i] → word[i-1]" },
    { name: "Long-range", desc: "Distant connections", color: "green", example: "end → start" },
  ]

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">3. Multi-Head Attention</h1>
        <p className="text-slate-400">
          A word has multiple relationships. "bank" connects to "river" (semantic), is
          a noun (syntactic), and is at position 1 (positional). Multiple heads capture
          all of these.
        </p>
      </div>

      {/* Why multiple heads */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <h2 className="font-semibold text-white mb-4">Why Multiple Heads?</h2>
        <div className="grid md:grid-cols-4 gap-4">
          {headDescriptions.map((head, i) => (
            <button
              key={i}
              onClick={() => setActiveHead(activeHead === i ? null : i)}
              className={`text-left p-4 rounded-lg border transition-all ${
                activeHead === i
                  ? `bg-${head.color}-900/30 border-${head.color}-500`
                  : "bg-slate-800 border-slate-700 hover:border-slate-600"
              }`}
            >
              <div className="flex items-center gap-2 mb-2">
                <div
                  className={`w-6 h-6 rounded flex items-center justify-center text-xs font-bold text-white`}
                  style={{
                    backgroundColor:
                      head.color === "blue"
                        ? "#3b82f6"
                        : head.color === "purple"
                          ? "#8b5cf6"
                          : head.color === "orange"
                            ? "#f97316"
                            : "#22c55e",
                  }}
                >
                  {i + 1}
                </div>
                <span className="font-medium text-white">{head.name}</span>
              </div>
              <p className="text-sm text-slate-400">{head.desc}</p>
              <p className="text-xs text-slate-500 mt-1 font-mono">{head.example}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Attention heatmaps */}
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
        {headPatterns.map((pattern, headIdx) => (
          <div
            key={headIdx}
            className={`bg-slate-900 border rounded-xl p-4 transition-all ${
              activeHead === null || activeHead === headIdx
                ? "border-slate-800 opacity-100"
                : "border-slate-800 opacity-30"
            }`}
          >
            <div className="flex items-center gap-2 mb-3">
              <div
                className="w-5 h-5 rounded text-xs font-bold text-white flex items-center justify-center"
                style={{
                  backgroundColor:
                    headDescriptions[headIdx].color === "blue"
                      ? "#3b82f6"
                      : headDescriptions[headIdx].color === "purple"
                        ? "#8b5cf6"
                        : headDescriptions[headIdx].color === "orange"
                          ? "#f97316"
                          : "#22c55e",
                }}
              >
                {headIdx + 1}
              </div>
              <span className="text-sm text-white">{headDescriptions[headIdx].name}</span>
            </div>
            <div className="grid grid-cols-5 gap-0.5">
              {pattern.map((row, i) =>
                row.map((weight, j) => (
                  <div
                    key={`${i}-${j}`}
                    className="aspect-square rounded-sm flex items-center justify-center text-[10px]"
                    style={{
                      backgroundColor: `rgba(59, 130, 246, ${weight})`,
                      color: weight > 0.4 ? "white" : "transparent",
                    }}
                    title={`${words[i]} → ${words[j]}: ${(weight * 100).toFixed(0)}%`}
                  >
                    {weight > 0.4 && (weight * 100).toFixed(0)}
                  </div>
                ))
              )}
            </div>
            <div className="flex justify-between mt-2 text-[10px] text-slate-500">
              <span>Q↓</span>
              <span>K→</span>
            </div>
          </div>
        ))}
      </div>

      {/* Word labels */}
      <div className="flex justify-center gap-2">
        {words.map((word, i) => (
          <div key={i} className="text-sm text-slate-400 font-mono px-2">
            {word}
          </div>
        ))}
      </div>

      {/* Concatenation visualization */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <h2 className="font-semibold text-white mb-4">Concatenate + Project</h2>
        <p className="text-sm text-slate-400 mb-4">
          Each head outputs a smaller vector (d_model / num_heads). We concatenate all
          heads and project back to d_model.
        </p>

        <div className="flex items-center justify-center gap-2 flex-wrap">
          {headDescriptions.map((head, i) => (
            <div
              key={i}
              className="h-16 w-12 rounded flex items-center justify-center text-white text-xs font-bold"
              style={{
                backgroundColor:
                  head.color === "blue"
                    ? "#3b82f6"
                    : head.color === "purple"
                      ? "#8b5cf6"
                      : head.color === "orange"
                        ? "#f97316"
                        : "#22c55e",
              }}
            >
              H{i + 1}
            </div>
          ))}
          <div className="text-2xl text-slate-500">→</div>
          <div className="h-16 w-48 bg-gradient-to-r from-blue-600 via-purple-600 to-green-600 rounded flex items-center justify-center text-white text-xs font-bold">
            Concatenated
          </div>
          <div className="text-2xl text-slate-500">→</div>
          <div className="h-16 w-20 bg-slate-700 rounded flex items-center justify-center text-white text-xs">
            W<sub>O</sub>
          </div>
          <div className="text-2xl text-slate-500">→</div>
          <div className="h-16 w-32 bg-gradient-to-r from-cyan-600 to-blue-600 rounded flex items-center justify-center text-white text-xs font-bold">
            Output
          </div>
        </div>

        <div className="mt-4 text-xs text-slate-500 text-center">
          MultiHead(Q,K,V) = Concat(head₁, ..., head_h) × W<sub>O</sub>
        </div>
      </div>

      {/* Dimension math */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <h2 className="font-semibold text-white mb-4">Dimension Math</h2>
        <div className="font-mono text-sm space-y-2">
          <div className="flex items-center gap-4">
            <span className="text-slate-400 w-32">d_model:</span>
            <span className="text-white">512</span>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-slate-400 w-32">num_heads:</span>
            <span className="text-white">8</span>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-slate-400 w-32">d_k (per head):</span>
            <span className="text-white">512 / 8 = 64</span>
          </div>
          <div className="flex items-center gap-4 pt-2 border-t border-slate-700">
            <span className="text-slate-400 w-32">Concat output:</span>
            <span className="text-white">8 × 64 = 512</span>
            <span className="text-slate-500">(back to d_model)</span>
          </div>
        </div>
      </div>

      {/* Key insight */}
      <div className="bg-gradient-to-r from-orange-900/30 to-red-900/30 border border-orange-800/50 rounded-xl p-6">
        <h2 className="font-semibold text-white mb-2">Key Insight</h2>
        <p className="text-slate-300">
          Each head is like a specialist examining the sentence differently. Head 1
          might focus on semantic meaning, Head 2 on grammar, Head 3 on position. The
          model learns WHICH relationships are useful during training. The final output
          combines insights from all perspectives.
        </p>
      </div>
    </div>
  )
}
