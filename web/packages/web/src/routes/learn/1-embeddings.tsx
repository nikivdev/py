import { createFileRoute } from "@tanstack/react-router"
import { useState, useMemo } from "react"

export const Route = createFileRoute("/learn/1-embeddings")({
  component: EmbeddingsLesson,
})

function EmbeddingsLesson() {
  const [selectedWord, setSelectedWord] = useState(1)
  const [showPositional, setShowPositional] = useState(true)
  const [dimension, setDimension] = useState(8)

  const words = ["The", "bank", "of", "the", "river"]

  // Simulated embeddings (seeded for consistency)
  const embeddings = useMemo(() => {
    const seed = (i: number, j: number) =>
      Math.sin(i * 12.9898 + j * 78.233) * 43758.5453 % 1
    return words.map((_, i) =>
      Array.from({ length: dimension }, (_, j) => seed(i * 100 + j, i) * 2 - 1)
    )
  }, [dimension])

  // Sinusoidal positional encoding
  const positionalEncoding = useMemo(() => {
    return words.map((_, pos) =>
      Array.from({ length: dimension }, (_, i) => {
        const divTerm = Math.pow(10000, (2 * Math.floor(i / 2)) / dimension)
        return i % 2 === 0
          ? Math.sin(pos / divTerm)
          : Math.cos(pos / divTerm)
      })
    )
  }, [dimension])

  const combined = useMemo(() => {
    return embeddings.map((emb, i) =>
      emb.map((v, j) => v + (showPositional ? positionalEncoding[i][j] : 0))
    )
  }, [embeddings, positionalEncoding, showPositional])

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">
          1. Embeddings + Positional Encoding
        </h1>
        <p className="text-slate-400">
          Transformers see all words at once. Without position info, "Man bites dog"
          and "Dog bites man" look identical!
        </p>
      </div>

      {/* Interactive sentence */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <h2 className="text-sm font-medium text-slate-500 mb-4">Input Sentence</h2>
        <div className="flex gap-2 flex-wrap">
          {words.map((word, i) => (
            <button
              key={i}
              onClick={() => setSelectedWord(i)}
              className={`px-4 py-2 rounded-lg font-mono text-lg transition-all ${
                selectedWord === i
                  ? "bg-blue-600 text-white scale-105"
                  : "bg-slate-800 text-slate-300 hover:bg-slate-700"
              }`}
            >
              {word}
              <span className="ml-2 text-xs opacity-50">[{i}]</span>
            </button>
          ))}
        </div>
      </div>

      {/* Vector visualization */}
      <div className="grid lg:grid-cols-3 gap-4">
        <VectorCard
          title="Word Embedding"
          subtitle="Semantic meaning"
          values={embeddings[selectedWord]}
          color="blue"
          description={`What "${words[selectedWord]}" means`}
        />
        <div className="flex items-center justify-center text-4xl text-slate-600">+</div>
        <VectorCard
          title="Positional Encoding"
          subtitle={`Position ${selectedWord}`}
          values={positionalEncoding[selectedWord]}
          color="purple"
          description="Where in the sequence"
          dimmed={!showPositional}
        />
      </div>

      <div className="flex justify-center">
        <div className="text-4xl text-slate-600">=</div>
      </div>

      <VectorCard
        title="Combined Input"
        subtitle="Ready for transformer"
        values={combined[selectedWord]}
        color="green"
        description="Meaning + Position"
        large
      />

      {/* Controls */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
        <div className="flex items-center justify-between">
          <label className="text-sm text-slate-400">Show Positional Encoding</label>
          <button
            onClick={() => setShowPositional(!showPositional)}
            className={`w-12 h-6 rounded-full transition-colors ${
              showPositional ? "bg-blue-600" : "bg-slate-700"
            }`}
          >
            <div
              className={`w-5 h-5 bg-white rounded-full transition-transform ${
                showPositional ? "translate-x-6" : "translate-x-0.5"
              }`}
            />
          </button>
        </div>
        <div className="flex items-center justify-between">
          <label className="text-sm text-slate-400">Vector Dimension</label>
          <select
            value={dimension}
            onChange={(e) => setDimension(Number(e.target.value))}
            className="bg-slate-800 border border-slate-700 rounded px-3 py-1 text-sm"
          >
            <option value={4}>4</option>
            <option value={8}>8</option>
            <option value={16}>16</option>
            <option value={32}>32</option>
          </select>
        </div>
      </div>

      {/* Sinusoidal pattern visualization */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <h2 className="font-semibold text-white mb-4">
          Sinusoidal Positional Encoding Pattern
        </h2>
        <p className="text-sm text-slate-400 mb-4">
          Each position gets a unique pattern of sin/cos waves at different frequencies.
          This allows the model to learn relative positions.
        </p>
        <div className="h-48 relative bg-slate-950 rounded-lg overflow-hidden">
          <svg viewBox="0 0 400 150" className="w-full h-full">
            {/* Grid */}
            {Array.from({ length: 5 }).map((_, i) => (
              <line
                key={i}
                x1={0}
                y1={30 + i * 30}
                x2={400}
                y2={30 + i * 30}
                stroke="#334155"
                strokeWidth={0.5}
              />
            ))}
            {/* Sin waves for different dimensions */}
            {[0, 2, 4, 6].map((dim, idx) => {
              const freq = Math.pow(10000, (2 * Math.floor(dim / 2)) / dimension)
              const color = ["#3b82f6", "#8b5cf6", "#ec4899", "#f97316"][idx]
              const points = Array.from({ length: 100 }, (_, x) => {
                const pos = x / 20
                const y = Math.sin(pos / freq) * 30 + 75
                return `${x * 4},${y}`
              }).join(" ")
              return (
                <polyline
                  key={dim}
                  points={points}
                  fill="none"
                  stroke={color}
                  strokeWidth={2}
                  opacity={0.7}
                />
              )
            })}
            {/* Position markers */}
            {words.map((_, pos) => (
              <g key={pos}>
                <line
                  x1={pos * 80 + 40}
                  y1={20}
                  x2={pos * 80 + 40}
                  y2={130}
                  stroke={pos === selectedWord ? "#3b82f6" : "#475569"}
                  strokeWidth={pos === selectedWord ? 2 : 1}
                  strokeDasharray={pos === selectedWord ? "0" : "4"}
                />
                <text
                  x={pos * 80 + 40}
                  y={145}
                  textAnchor="middle"
                  fill={pos === selectedWord ? "#3b82f6" : "#64748b"}
                  fontSize={12}
                >
                  {words[pos]}
                </text>
              </g>
            ))}
          </svg>
        </div>
        <p className="text-xs text-slate-500 mt-2">
          Different colors = different dimension frequencies. The vertical line shows
          where each word samples from these waves.
        </p>
      </div>

      {/* Key insight */}
      <div className="bg-gradient-to-r from-blue-900/30 to-purple-900/30 border border-blue-800/50 rounded-xl p-6">
        <h2 className="font-semibold text-white mb-2">Key Insight</h2>
        <p className="text-slate-300">
          The word embedding and positional encoding are <strong>added</strong>, not
          concatenated. In high-dimensional space, these vectors are nearly orthogonal,
          so the model can learn to separate semantic meaning from positional information.
        </p>
      </div>
    </div>
  )
}

function VectorCard({
  title,
  subtitle,
  values,
  color,
  description,
  dimmed = false,
  large = false,
}: {
  title: string
  subtitle: string
  values: number[]
  color: "blue" | "purple" | "green"
  description: string
  dimmed?: boolean
  large?: boolean
}) {
  const colorMap = {
    blue: "from-blue-500 to-cyan-500",
    purple: "from-purple-500 to-pink-500",
    green: "from-green-500 to-emerald-500",
  }

  return (
    <div
      className={`bg-slate-900 border border-slate-800 rounded-xl p-4 ${
        dimmed ? "opacity-40" : ""
      } ${large ? "lg:col-span-3" : ""}`}
    >
      <div className="flex items-center gap-2 mb-2">
        <div className={`w-3 h-3 rounded bg-gradient-to-r ${colorMap[color]}`} />
        <h3 className="font-medium text-white">{title}</h3>
        <span className="text-xs text-slate-500">{subtitle}</span>
      </div>
      <p className="text-xs text-slate-400 mb-3">{description}</p>
      <div className={`flex gap-1 flex-wrap ${large ? "justify-center" : ""}`}>
        {values.map((v, i) => (
          <div
            key={i}
            className="relative group"
            style={{ width: large ? 40 : 32, height: large ? 40 : 32 }}
          >
            <div
              className={`absolute inset-0 rounded transition-all ${
                v > 0 ? "bg-blue-500" : "bg-red-500"
              }`}
              style={{ opacity: Math.min(Math.abs(v) * 0.5, 1) }}
            />
            <div className="absolute inset-0 flex items-center justify-center text-xs font-mono text-white/80">
              {v.toFixed(1)}
            </div>
            <div className="absolute -top-8 left-1/2 -translate-x-1/2 bg-slate-800 px-2 py-1 rounded text-xs opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
              dim[{i}]: {v.toFixed(4)}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
