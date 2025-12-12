import { createFileRoute } from "@tanstack/react-router"
import { useState, useMemo } from "react"

export const Route = createFileRoute("/learn/2-attention")({
  component: AttentionLesson,
})

function AttentionLesson() {
  const [selectedQuery, setSelectedQuery] = useState(1) // "bank"
  const [showScaled, setShowScaled] = useState(true)

  const words = ["The", "bank", "of", "the", "river"]
  const d_k = 4

  // Simulated Q, K, V matrices (small for visualization)
  const seed = (i: number, j: number, offset: number) =>
    (Math.sin((i + offset) * 12.9898 + j * 78.233) * 43758.5453 % 1) * 2 - 1

  const Q = useMemo(
    () => words.map((_, i) => Array.from({ length: d_k }, (_, j) => seed(i, j, 0))),
    []
  )
  const K = useMemo(
    () => words.map((_, i) => Array.from({ length: d_k }, (_, j) => seed(i, j, 100))),
    []
  )
  const V = useMemo(
    () => words.map((_, i) => Array.from({ length: d_k }, (_, j) => seed(i, j, 200))),
    []
  )

  // Compute attention scores for selected query
  const rawScores = useMemo(() => {
    const q = Q[selectedQuery]
    return K.map((k) => q.reduce((sum, qv, i) => sum + qv * k[i], 0))
  }, [Q, K, selectedQuery])

  const scaledScores = useMemo(
    () => rawScores.map((s) => s / Math.sqrt(d_k)),
    [rawScores]
  )

  const scores = showScaled ? scaledScores : rawScores

  // Softmax
  const attentionWeights = useMemo(() => {
    const maxScore = Math.max(...scores)
    const expScores = scores.map((s) => Math.exp(s - maxScore))
    const sumExp = expScores.reduce((a, b) => a + b, 0)
    return expScores.map((e) => e / sumExp)
  }, [scores])

  // Weighted sum of values
  const output = useMemo(() => {
    return Array.from({ length: d_k }, (_, dim) =>
      V.reduce((sum, v, i) => sum + attentionWeights[i] * v[dim], 0)
    )
  }, [V, attentionWeights])

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">
          2. Self-Attention (Query, Key, Value)
        </h1>
        <p className="text-slate-400">
          The heart of transformers. How "bank" learns it means "riverbank" not
          "financial bank" from context.
        </p>
      </div>

      {/* The analogy */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <h2 className="font-semibold text-white mb-4">The File Cabinet Analogy</h2>
        <div className="grid md:grid-cols-3 gap-4 text-sm">
          <div className="bg-blue-900/20 border border-blue-800/50 rounded-lg p-4">
            <div className="text-blue-400 font-medium mb-1">Query (Q)</div>
            <p className="text-slate-400">The sticky note with what you're looking for</p>
          </div>
          <div className="bg-purple-900/20 border border-purple-800/50 rounded-lg p-4">
            <div className="text-purple-400 font-medium mb-1">Key (K)</div>
            <p className="text-slate-400">The label on folder tabs (for matching)</p>
          </div>
          <div className="bg-green-900/20 border border-green-800/50 rounded-lg p-4">
            <div className="text-green-400 font-medium mb-1">Value (V)</div>
            <p className="text-slate-400">The actual papers inside the folder</p>
          </div>
        </div>
      </div>

      {/* Select query word */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <h2 className="text-sm font-medium text-slate-500 mb-4">
          Select Query Word (what's looking for context)
        </h2>
        <div className="flex gap-2 flex-wrap">
          {words.map((word, i) => (
            <button
              key={i}
              onClick={() => setSelectedQuery(i)}
              className={`px-4 py-2 rounded-lg font-mono transition-all ${
                selectedQuery === i
                  ? "bg-blue-600 text-white"
                  : "bg-slate-800 text-slate-300 hover:bg-slate-700"
              }`}
            >
              {word}
            </button>
          ))}
        </div>
        <p className="text-sm text-slate-400 mt-3">
          Q<sub>{words[selectedQuery]}</sub> is asking: "What other words should I
          attend to?"
        </p>
      </div>

      {/* Q, K matrices */}
      <div className="grid lg:grid-cols-2 gap-4">
        <MatrixVis
          title={`Query: "${words[selectedQuery]}"`}
          data={[Q[selectedQuery]]}
          rowLabels={[words[selectedQuery]]}
          colLabels={["d₀", "d₁", "d₂", "d₃"]}
          color="blue"
          highlightRow={0}
        />
        <MatrixVis
          title="All Keys"
          data={K}
          rowLabels={words}
          colLabels={["d₀", "d₁", "d₂", "d₃"]}
          color="purple"
        />
      </div>

      {/* Dot product visualization */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="font-semibold text-white">
            Step 1: Q·K<sup>T</sup> (Similarity Scores)
          </h2>
          <label className="flex items-center gap-2 text-sm">
            <span className="text-slate-400">Scale by √d_k</span>
            <button
              onClick={() => setShowScaled(!showScaled)}
              className={`w-10 h-5 rounded-full transition-colors ${
                showScaled ? "bg-blue-600" : "bg-slate-700"
              }`}
            >
              <div
                className={`w-4 h-4 bg-white rounded-full transition-transform ${
                  showScaled ? "translate-x-5" : "translate-x-0.5"
                }`}
              />
            </button>
          </label>
        </div>

        <div className="flex items-center gap-4 flex-wrap">
          {words.map((word, i) => {
            const score = scores[i]
            const weight = attentionWeights[i]
            return (
              <div
                key={i}
                className={`bg-slate-800 rounded-lg p-3 text-center transition-all ${
                  i === selectedQuery ? "ring-2 ring-blue-500" : ""
                }`}
                style={{
                  transform: `scale(${0.9 + weight * 0.3})`,
                }}
              >
                <div className="text-sm text-slate-400">{word}</div>
                <div className="font-mono text-white">{score.toFixed(2)}</div>
                <div className="text-xs text-slate-500">
                  Q<sub>{words[selectedQuery]}</sub>·K<sub>{word}</sub>
                </div>
              </div>
            )
          })}
        </div>

        {!showScaled && (
          <p className="text-xs text-yellow-400 mt-3">
            ⚠️ Without scaling, large dot products push softmax to extremes
          </p>
        )}
      </div>

      {/* Softmax attention weights */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <h2 className="font-semibold text-white mb-4">
          Step 2: Softmax → Attention Weights
        </h2>
        <p className="text-sm text-slate-400 mb-4">
          Scores converted to probabilities that sum to 1
        </p>

        <div className="flex items-end gap-2 h-32">
          {words.map((word, i) => (
            <div key={i} className="flex-1 flex flex-col items-center">
              <div className="text-xs text-white mb-1">
                {(attentionWeights[i] * 100).toFixed(1)}%
              </div>
              <div
                className="w-full bg-gradient-to-t from-blue-600 to-cyan-400 rounded-t transition-all"
                style={{ height: `${attentionWeights[i] * 100}%` }}
              />
              <div className="text-xs text-slate-400 mt-2">{word}</div>
            </div>
          ))}
        </div>

        <p className="text-sm text-slate-400 mt-4">
          "{words[selectedQuery]}" attends most strongly to{" "}
          <strong className="text-white">
            "{words[attentionWeights.indexOf(Math.max(...attentionWeights))]}"
          </strong>
        </p>
      </div>

      {/* Value weighted sum */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <h2 className="font-semibold text-white mb-4">
          Step 3: Weighted Sum of Values
        </h2>
        <p className="text-sm text-slate-400 mb-4">
          Output = Σ (attention_weight × Value)
        </p>

        <div className="grid lg:grid-cols-2 gap-4">
          <MatrixVis
            title="Values (information content)"
            data={V}
            rowLabels={words}
            colLabels={["d₀", "d₁", "d₂", "d₃"]}
            color="green"
            rowWeights={attentionWeights}
          />
          <div className="bg-slate-800 rounded-lg p-4">
            <h3 className="text-sm text-slate-400 mb-3">
              New vector for "{words[selectedQuery]}"
            </h3>
            <div className="flex gap-2">
              {output.map((v, i) => (
                <div
                  key={i}
                  className="flex-1 bg-gradient-to-b from-green-600 to-emerald-700 rounded p-2 text-center"
                >
                  <div className="text-xs text-green-200">d{i}</div>
                  <div className="font-mono text-white">{v.toFixed(2)}</div>
                </div>
              ))}
            </div>
            <p className="text-xs text-slate-400 mt-3">
              This vector now encodes "{words[selectedQuery]}" in context of the
              other words it attended to!
            </p>
          </div>
        </div>
      </div>

      {/* Attention matrix visualization */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <h2 className="font-semibold text-white mb-4">Full Attention Matrix</h2>
        <p className="text-sm text-slate-400 mb-4">
          Row i shows what word i attends to. Column j shows how much word i attends
          to word j.
        </p>
        <AttentionHeatmap words={words} Q={Q} K={K} d_k={d_k} selectedQuery={selectedQuery} />
      </div>

      {/* Key insight */}
      <div className="bg-gradient-to-r from-purple-900/30 to-pink-900/30 border border-purple-800/50 rounded-xl p-6">
        <h2 className="font-semibold text-white mb-2">The Magic</h2>
        <p className="text-slate-300">
          The weight matrices W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub> are{" "}
          <strong>learned</strong>. Through training, the model learns what to look
          for (Q), how to be found (K), and what information to share (V). The
          attention mechanism provides the capacity; training fills it with knowledge.
        </p>
      </div>
    </div>
  )
}

function MatrixVis({
  title,
  data,
  rowLabels,
  colLabels,
  color,
  highlightRow,
  rowWeights,
}: {
  title: string
  data: number[][]
  rowLabels: string[]
  colLabels: string[]
  color: "blue" | "purple" | "green"
  highlightRow?: number
  rowWeights?: number[]
}) {
  const colorMap = {
    blue: "bg-blue-500",
    purple: "bg-purple-500",
    green: "bg-green-500",
  }

  return (
    <div className="bg-slate-800 rounded-lg p-4">
      <h3 className="text-sm text-slate-400 mb-3">{title}</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr>
              <th />
              {colLabels.map((col, i) => (
                <th key={i} className="text-slate-500 font-normal p-1">
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row, i) => (
              <tr
                key={i}
                className={highlightRow === i ? "bg-slate-700/50" : ""}
                style={rowWeights ? { opacity: 0.3 + rowWeights[i] * 0.7 } : {}}
              >
                <td className="text-slate-400 pr-2 font-mono">{rowLabels[i]}</td>
                {row.map((val, j) => (
                  <td key={j} className="p-1">
                    <div
                      className={`${colorMap[color]} rounded px-1 py-0.5 text-center text-white`}
                      style={{ opacity: 0.3 + Math.abs(val) * 0.5 }}
                    >
                      {val.toFixed(1)}
                    </div>
                  </td>
                ))}
                {rowWeights && (
                  <td className="pl-2 text-slate-500">
                    ×{(rowWeights[i] * 100).toFixed(0)}%
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function AttentionHeatmap({
  words,
  Q,
  K,
  d_k,
  selectedQuery,
}: {
  words: string[]
  Q: number[][]
  K: number[][]
  d_k: number
  selectedQuery: number
}) {
  // Compute full attention matrix
  const attentionMatrix = useMemo(() => {
    return Q.map((q) => {
      const scores = K.map((k) => q.reduce((sum, qv, i) => sum + qv * k[i], 0) / Math.sqrt(d_k))
      const maxScore = Math.max(...scores)
      const expScores = scores.map((s) => Math.exp(s - maxScore))
      const sumExp = expScores.reduce((a, b) => a + b, 0)
      return expScores.map((e) => e / sumExp)
    })
  }, [Q, K, d_k])

  return (
    <div className="overflow-x-auto">
      <table className="mx-auto">
        <thead>
          <tr>
            <th className="p-2" />
            {words.map((word, i) => (
              <th key={i} className="p-2 text-sm text-slate-400 font-normal">
                {word}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {words.map((word, i) => (
            <tr key={i}>
              <td className="p-2 text-sm text-slate-400">{word}</td>
              {attentionMatrix[i].map((weight, j) => (
                <td key={j} className="p-1">
                  <div
                    className={`w-12 h-12 rounded flex items-center justify-center text-xs font-mono transition-all ${
                      i === selectedQuery ? "ring-2 ring-blue-500" : ""
                    }`}
                    style={{
                      backgroundColor: `rgba(59, 130, 246, ${weight})`,
                      color: weight > 0.5 ? "white" : "#94a3b8",
                    }}
                  >
                    {(weight * 100).toFixed(0)}%
                  </div>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
