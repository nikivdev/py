import { createFileRoute } from "@tanstack/react-router"
import { useState, useEffect, useCallback } from "react"

export const Route = createFileRoute("/learn/7-training")({
  component: TrainingLesson,
})

function TrainingLesson() {
  const [isTraining, setIsTraining] = useState(false)
  const [epoch, setEpoch] = useState(0)
  const [loss, setLoss] = useState(3.0)
  const [losses, setLosses] = useState<number[]>([3.0])
  const [weights, setWeights] = useState<number[][]>(
    Array.from({ length: 4 }, () =>
      Array.from({ length: 4 }, () => Math.random() * 2 - 1)
    )
  )
  const [predictions, setPredictions] = useState<number[]>([0, 0, 0, 0])

  const targetPattern = [2, 3, 4, 5] // Learn: output = input + 1
  const input = [1, 2, 3, 4]

  const trainStep = useCallback(() => {
    // Simulate training: weights converge toward identity + 1
    setWeights((prev) =>
      prev.map((row, i) =>
        row.map((w, j) => {
          const target = i === j ? 1 : 0
          const lr = 0.1
          return w + lr * (target - w) + (Math.random() - 0.5) * 0.05
        })
      )
    )

    // Compute predictions based on weights
    const newPreds = input.map((inp, i) => {
      const sum = weights[i].reduce((acc, w, j) => acc + w * input[j], 0)
      return Math.round(sum + 1) // +1 bias
    })
    setPredictions(newPreds)

    // Compute loss
    const newLoss = Math.max(
      0.01,
      targetPattern.reduce((acc, t, i) => acc + Math.abs(t - newPreds[i]), 0) / 4 +
        Math.random() * 0.1
    )

    setLoss(newLoss)
    setLosses((prev) => [...prev.slice(-49), newLoss])
    setEpoch((prev) => prev + 1)
  }, [weights])

  useEffect(() => {
    if (!isTraining) return
    const interval = setInterval(trainStep, 100)
    return () => clearInterval(interval)
  }, [isTraining, trainStep])

  const reset = () => {
    setIsTraining(false)
    setEpoch(0)
    setLoss(3.0)
    setLosses([3.0])
    setWeights(
      Array.from({ length: 4 }, () =>
        Array.from({ length: 4 }, () => Math.random() * 2 - 1)
      )
    )
    setPredictions([0, 0, 0, 0])
  }

  const accuracy =
    predictions.filter((p, i) => p === targetPattern[i]).length / 4

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">7. Training Demo</h1>
        <p className="text-slate-400">
          Watch how weights learn through backpropagation. The model learns to predict:
          next_token = current_token + 1
        </p>
      </div>

      {/* Task description */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <h2 className="font-semibold text-white mb-4">The Task</h2>
        <div className="flex items-center gap-8 justify-center">
          <div className="text-center">
            <div className="text-sm text-slate-400 mb-2">Input</div>
            <div className="flex gap-2">
              {input.map((v, i) => (
                <div
                  key={i}
                  className="w-12 h-12 bg-blue-600 rounded flex items-center justify-center text-white font-mono"
                >
                  {v}
                </div>
              ))}
            </div>
          </div>
          <div className="text-3xl text-slate-500">→</div>
          <div className="text-center">
            <div className="text-sm text-slate-400 mb-2">Target</div>
            <div className="flex gap-2">
              {targetPattern.map((v, i) => (
                <div
                  key={i}
                  className="w-12 h-12 bg-green-600 rounded flex items-center justify-center text-white font-mono"
                >
                  {v}
                </div>
              ))}
            </div>
          </div>
        </div>
        <p className="text-center text-sm text-slate-400 mt-4">
          The model must learn: output[i] = input[i] + 1
        </p>
      </div>

      {/* Training controls */}
      <div className="flex justify-center gap-4">
        <button
          onClick={() => setIsTraining(!isTraining)}
          className={`px-6 py-3 rounded-lg font-medium transition-all ${
            isTraining
              ? "bg-red-600 hover:bg-red-700 text-white"
              : "bg-green-600 hover:bg-green-700 text-white"
          }`}
        >
          {isTraining ? "⏸ Pause" : "▶ Train"}
        </button>
        <button
          onClick={trainStep}
          disabled={isTraining}
          className="px-6 py-3 rounded-lg font-medium bg-blue-600 hover:bg-blue-700 text-white disabled:opacity-50"
        >
          Step
        </button>
        <button
          onClick={reset}
          className="px-6 py-3 rounded-lg font-medium bg-slate-700 hover:bg-slate-600 text-white"
        >
          Reset
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 text-center">
          <div className="text-3xl font-mono text-white">{epoch}</div>
          <div className="text-sm text-slate-400">Epoch</div>
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 text-center">
          <div className="text-3xl font-mono text-white">{loss.toFixed(3)}</div>
          <div className="text-sm text-slate-400">Loss</div>
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 text-center">
          <div
            className={`text-3xl font-mono ${accuracy === 1 ? "text-green-400" : "text-white"}`}
          >
            {(accuracy * 100).toFixed(0)}%
          </div>
          <div className="text-sm text-slate-400">Accuracy</div>
        </div>
      </div>

      {/* Predictions */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <h2 className="font-semibold text-white mb-4">Current Predictions</h2>
        <div className="flex items-center gap-8 justify-center">
          <div className="text-center">
            <div className="text-sm text-slate-400 mb-2">Input</div>
            <div className="flex gap-2">
              {input.map((v, i) => (
                <div
                  key={i}
                  className="w-12 h-12 bg-blue-600/50 rounded flex items-center justify-center text-white font-mono"
                >
                  {v}
                </div>
              ))}
            </div>
          </div>
          <div className="text-3xl text-slate-500">→</div>
          <div className="text-center">
            <div className="text-sm text-slate-400 mb-2">Predicted</div>
            <div className="flex gap-2">
              {predictions.map((v, i) => (
                <div
                  key={i}
                  className={`w-12 h-12 rounded flex items-center justify-center text-white font-mono transition-all ${
                    v === targetPattern[i]
                      ? "bg-green-600 ring-2 ring-green-400"
                      : "bg-red-600/50"
                  }`}
                >
                  {v}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Loss curve */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <h2 className="font-semibold text-white mb-4">Loss Curve</h2>
        <div className="h-40 flex items-end gap-0.5">
          {losses.map((l, i) => (
            <div
              key={i}
              className="flex-1 bg-gradient-to-t from-blue-600 to-cyan-400 rounded-t transition-all"
              style={{ height: `${Math.min(l / 3, 1) * 100}%` }}
            />
          ))}
        </div>
        <div className="flex justify-between text-xs text-slate-500 mt-2">
          <span>Epoch {Math.max(0, epoch - 50)}</span>
          <span>Epoch {epoch}</span>
        </div>
      </div>

      {/* Weight visualization */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <h2 className="font-semibold text-white mb-4">Weight Matrix (Learning)</h2>
        <p className="text-sm text-slate-400 mb-4">
          Weights converge toward an identity-like matrix (diagonal = 1) as the model
          learns to pass through the input and add 1.
        </p>
        <div className="flex justify-center">
          <div className="grid grid-cols-4 gap-1">
            {weights.map((row, i) =>
              row.map((w, j) => (
                <div
                  key={`${i}-${j}`}
                  className="w-16 h-16 rounded flex items-center justify-center text-xs font-mono transition-all"
                  style={{
                    backgroundColor:
                      w > 0
                        ? `rgba(34, 197, 94, ${Math.min(Math.abs(w), 1)})`
                        : `rgba(239, 68, 68, ${Math.min(Math.abs(w), 1)})`,
                    color: Math.abs(w) > 0.5 ? "white" : "#94a3b8",
                  }}
                >
                  {w.toFixed(2)}
                </div>
              ))
            )}
          </div>
        </div>
        <p className="text-xs text-slate-500 text-center mt-4">
          Diagonal elements should approach 1.0 (green) for identity transform
        </p>
      </div>

      {/* What's being learned */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <h2 className="font-semibold text-white mb-4">What's Being Learned</h2>
        <div className="grid md:grid-cols-3 gap-4 text-sm">
          <div className="bg-blue-900/20 border border-blue-800/50 rounded-lg p-4">
            <div className="text-blue-400 font-medium mb-2">Attention Weights</div>
            <p className="text-slate-400">
              W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub> learn what features to look
              for and what to pass along.
            </p>
          </div>
          <div className="bg-green-900/20 border border-green-800/50 rounded-lg p-4">
            <div className="text-green-400 font-medium mb-2">FFN Weights</div>
            <p className="text-slate-400">
              W<sub>1</sub>, W<sub>2</sub> learn transformations and "facts" about the
              task.
            </p>
          </div>
          <div className="bg-purple-900/20 border border-purple-800/50 rounded-lg p-4">
            <div className="text-purple-400 font-medium mb-2">Output Projection</div>
            <p className="text-slate-400">
              Maps final hidden state to vocabulary probabilities.
            </p>
          </div>
        </div>
      </div>

      {/* Gradient descent visualization */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <h2 className="font-semibold text-white mb-4">The Learning Loop</h2>
        <div className="flex items-center justify-center gap-4 flex-wrap">
          {[
            { label: "Forward Pass", desc: "Compute predictions" },
            { label: "Compute Loss", desc: "How wrong are we?" },
            { label: "Backward Pass", desc: "∂Loss/∂weights" },
            { label: "Update Weights", desc: "w = w - lr × gradient" },
          ].map((step, i) => (
            <div key={i} className="flex items-center gap-2">
              <div
                className={`w-32 h-20 rounded-lg p-3 text-center ${
                  epoch % 4 === i && isTraining
                    ? "bg-blue-600 ring-2 ring-blue-400"
                    : "bg-slate-800"
                }`}
              >
                <div className="text-white text-sm font-medium">{step.label}</div>
                <div className="text-xs text-slate-400 mt-1">{step.desc}</div>
              </div>
              {i < 3 && <div className="text-slate-500">→</div>}
            </div>
          ))}
        </div>
        <div className="text-center mt-4">
          <div className="text-slate-500">↩ Repeat</div>
        </div>
      </div>

      {/* Key insight */}
      <div className="bg-gradient-to-r from-rose-900/30 to-pink-900/30 border border-rose-800/50 rounded-xl p-6">
        <h2 className="font-semibold text-white mb-2">The Magic</h2>
        <p className="text-slate-300">
          The model doesn't have hardcoded rules. ALL knowledge comes from adjusting
          weights to minimize loss. The attention mechanism provides CAPACITY to learn
          relationships. Training provides KNOWLEDGE. Scale (data + parameters)
          determines QUALITY. Everything emerges from: ∂Loss/∂weights → update weights.
        </p>
      </div>
    </div>
  )
}
