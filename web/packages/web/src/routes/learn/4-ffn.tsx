import { createFileRoute } from "@tanstack/react-router"
import { useState, useMemo } from "react"

export const Route = createFileRoute("/learn/4-ffn")({
  component: FFNLesson,
})

function FFNLesson() {
  const [inputValues, setInputValues] = useState([0.5, -0.3, 0.8, -0.2])
  const d_model = 4
  const d_ff = 16

  // Simulated FFN weights
  const W1 = useMemo(
    () =>
      Array.from({ length: d_ff }, (_, i) =>
        Array.from(
          { length: d_model },
          (_, j) => (Math.sin(i * 12.9898 + j * 78.233) * 43758.5453 % 1) * 2 - 1
        )
      ),
    []
  )

  const W2 = useMemo(
    () =>
      Array.from({ length: d_model }, (_, i) =>
        Array.from(
          { length: d_ff },
          (_, j) => (Math.sin((i + 100) * 12.9898 + j * 78.233) * 43758.5453 % 1) * 2 - 1
        )
      ),
    []
  )

  // Forward pass
  const hidden = useMemo(() => {
    return W1.map((row) =>
      row.reduce((sum, w, i) => sum + w * inputValues[i], 0)
    )
  }, [W1, inputValues])

  const afterRelu = useMemo(() => hidden.map((h) => Math.max(0, h)), [hidden])

  const output = useMemo(() => {
    return W2.map((row) => row.reduce((sum, w, i) => sum + w * afterRelu[i], 0))
  }, [W2, afterRelu])

  const sparsity = afterRelu.filter((v) => v === 0).length / afterRelu.length

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">
          4. Feed-Forward Network (FFN)
        </h1>
        <p className="text-slate-400">
          After attention gathers context, each word goes through an FFN individually.
          Attention = "look around". FFN = "think and process".
        </p>
      </div>

      {/* Two phases */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-blue-900/20 border border-blue-800/50 rounded-xl p-6">
          <h2 className="text-blue-400 font-medium mb-2">Multi-Head Attention</h2>
          <p className="text-slate-400 text-sm">
            "Look around and gather context from other words"
          </p>
        </div>
        <div className="bg-green-900/20 border border-green-800/50 rounded-xl p-6">
          <h2 className="text-green-400 font-medium mb-2">Feed-Forward Network</h2>
          <p className="text-slate-400 text-sm">
            "Think and process" - where knowledge is stored
          </p>
        </div>
      </div>

      {/* Architecture visualization */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <h2 className="font-semibold text-white mb-4">FFN Architecture</h2>
        <p className="text-sm text-slate-400 mb-6">
          FFN(x) = ReLU(x × W₁ + b₁) × W₂ + b₂
        </p>

        <div className="flex items-center justify-center gap-4 overflow-x-auto py-4">
          {/* Input */}
          <div className="flex flex-col gap-1">
            <div className="text-xs text-slate-500 text-center mb-1">Input</div>
            {inputValues.map((v, i) => (
              <input
                key={i}
                type="range"
                min="-1"
                max="1"
                step="0.1"
                value={v}
                onChange={(e) => {
                  const newValues = [...inputValues]
                  newValues[i] = parseFloat(e.target.value)
                  setInputValues(newValues)
                }}
                className="w-16 h-8 bg-blue-600 rounded cursor-pointer"
                style={{
                  background: `linear-gradient(to right, ${v < 0 ? "#ef4444" : "#3b82f6"} ${50 + v * 50}%, #334155 ${50 + v * 50}%)`,
                }}
              />
            ))}
            <div className="text-xs text-slate-500 text-center">d={d_model}</div>
          </div>

          <div className="text-slate-500">→</div>

          {/* W1 */}
          <div className="bg-slate-800 rounded p-2">
            <div className="text-xs text-slate-400 mb-1">W₁</div>
            <div className="text-xs text-slate-500">{d_model}×{d_ff}</div>
          </div>

          <div className="text-slate-500">→</div>

          {/* Hidden layer */}
          <div className="flex flex-col gap-0.5">
            <div className="text-xs text-slate-500 text-center mb-1">Hidden</div>
            <div className="flex flex-wrap w-24 gap-0.5">
              {afterRelu.map((v, i) => (
                <div
                  key={i}
                  className="w-5 h-5 rounded-sm transition-all"
                  style={{
                    backgroundColor: v > 0 ? `rgba(34, 197, 94, ${Math.min(v, 1)})` : "#1e293b",
                  }}
                  title={`neuron ${i}: ${v.toFixed(2)}`}
                />
              ))}
            </div>
            <div className="text-xs text-slate-500 text-center">d={d_ff}</div>
            <div className="text-xs text-yellow-500 text-center">
              {(sparsity * 100).toFixed(0)}% zero
            </div>
          </div>

          <div className="text-slate-500">→</div>

          {/* ReLU */}
          <div className="bg-yellow-900/30 border border-yellow-800/50 rounded p-2">
            <div className="text-xs text-yellow-400">ReLU</div>
          </div>

          <div className="text-slate-500">→</div>

          {/* W2 */}
          <div className="bg-slate-800 rounded p-2">
            <div className="text-xs text-slate-400 mb-1">W₂</div>
            <div className="text-xs text-slate-500">{d_ff}×{d_model}</div>
          </div>

          <div className="text-slate-500">→</div>

          {/* Output */}
          <div className="flex flex-col gap-1">
            <div className="text-xs text-slate-500 text-center mb-1">Output</div>
            {output.map((v, i) => (
              <div
                key={i}
                className="w-16 h-8 rounded flex items-center justify-center text-xs font-mono text-white"
                style={{
                  backgroundColor: v > 0 ? `rgba(34, 197, 94, ${Math.min(Math.abs(v) * 0.3, 0.8)})` : `rgba(239, 68, 68, ${Math.min(Math.abs(v) * 0.3, 0.8)})`,
                }}
              >
                {v.toFixed(2)}
              </div>
            ))}
            <div className="text-xs text-slate-500 text-center">d={d_model}</div>
          </div>
        </div>
      </div>

      {/* Expansion explanation */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <h2 className="font-semibold text-white mb-4">Why Expand Then Compress?</h2>
        <div className="space-y-4">
          <div className="flex items-start gap-4">
            <div className="w-8 h-8 bg-blue-600 rounded flex items-center justify-center text-white text-sm flex-shrink-0">
              1
            </div>
            <div>
              <p className="text-white">
                <strong>Expand</strong>: d_model → d_ff (usually 4×)
              </p>
              <p className="text-sm text-slate-400">
                More capacity to learn complex functions. Each hidden neuron can detect
                a different pattern.
              </p>
            </div>
          </div>
          <div className="flex items-start gap-4">
            <div className="w-8 h-8 bg-yellow-600 rounded flex items-center justify-center text-white text-sm flex-shrink-0">
              2
            </div>
            <div>
              <p className="text-white">
                <strong>ReLU</strong>: max(0, x)
              </p>
              <p className="text-sm text-slate-400">
                Creates sparsity - most neurons are zero. Only relevant features
                activate.
              </p>
            </div>
          </div>
          <div className="flex items-start gap-4">
            <div className="w-8 h-8 bg-green-600 rounded flex items-center justify-center text-white text-sm flex-shrink-0">
              3
            </div>
            <div>
              <p className="text-white">
                <strong>Compress</strong>: d_ff → d_model
              </p>
              <p className="text-sm text-slate-400">
                Combine detected features back into a meaningful representation.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Knowledge storage */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <h2 className="font-semibold text-white mb-4">FFN as Knowledge Bank</h2>
        <p className="text-sm text-slate-400 mb-4">
          The FFN weights store factual knowledge learned during training:
        </p>
        <div className="grid md:grid-cols-2 gap-4 text-sm">
          <div className="bg-slate-800 rounded p-3">
            <div className="text-green-400 font-mono mb-1">"Paris is the capital of..."</div>
            <div className="text-slate-400">
              → FFN weights encode the mapping to "France"
            </div>
          </div>
          <div className="bg-slate-800 rounded p-3">
            <div className="text-green-400 font-mono mb-1">"Water boils at..."</div>
            <div className="text-slate-400">→ FFN weights encode "100°C"</div>
          </div>
        </div>
      </div>

      {/* Modern variants */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <h2 className="font-semibold text-white mb-4">Modern Variants</h2>
        <div className="grid md:grid-cols-2 gap-4 text-sm">
          <div className="space-y-2">
            <div className="text-slate-400">Original (2017)</div>
            <code className="text-blue-400 text-xs">
              FFN(x) = ReLU(xW₁)W₂
            </code>
          </div>
          <div className="space-y-2">
            <div className="text-slate-400">GeGLU (LLaMA, PaLM)</div>
            <code className="text-green-400 text-xs">
              FFN(x) = (GELU(xW_gate) * xW_up)W_down
            </code>
            <p className="text-slate-500 text-xs">
              Gate controls information flow. Better gradients.
            </p>
          </div>
        </div>
      </div>

      {/* Key insight */}
      <div className="bg-gradient-to-r from-green-900/30 to-emerald-900/30 border border-green-800/50 rounded-xl p-6">
        <h2 className="font-semibold text-white mb-2">Key Insight</h2>
        <p className="text-slate-300">
          The FFN is applied <strong>independently</strong> to each position - no
          interaction between words (that's attention's job). But the same weights are
          shared across all positions. This is where the model stores "knowledge" -
          facts and transformations learned from training data.
        </p>
      </div>
    </div>
  )
}
