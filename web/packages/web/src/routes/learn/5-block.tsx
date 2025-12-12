import { createFileRoute } from "@tanstack/react-router"
import { useState } from "react"

export const Route = createFileRoute("/learn/5-block")({
  component: BlockLesson,
})

function BlockLesson() {
  const [showResidual, setShowResidual] = useState(true)
  const [preNorm, setPreNorm] = useState(true)
  const [step, setStep] = useState(0)

  const steps = [
    { name: "Input x", color: "#3b82f6" },
    { name: "LayerNorm", color: "#8b5cf6" },
    { name: "Multi-Head Attention", color: "#ec4899" },
    { name: "Add (Residual)", color: "#f97316" },
    { name: "LayerNorm", color: "#8b5cf6" },
    { name: "FFN", color: "#22c55e" },
    { name: "Add (Residual)", color: "#f97316" },
    { name: "Output", color: "#06b6d4" },
  ]

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">
          5. Transformer Block
        </h1>
        <p className="text-slate-400">
          Residual connections and Layer Normalization - the engineering secret sauce
          that allows transformers to be deep.
        </p>
      </div>

      {/* Block diagram */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="font-semibold text-white">Transformer Block</h2>
          <div className="flex gap-4">
            <label className="flex items-center gap-2 text-sm">
              <span className="text-slate-400">Pre-LN</span>
              <button
                onClick={() => setPreNorm(!preNorm)}
                className={`w-10 h-5 rounded-full transition-colors ${
                  preNorm ? "bg-blue-600" : "bg-slate-700"
                }`}
              >
                <div
                  className={`w-4 h-4 bg-white rounded-full transition-transform ${
                    preNorm ? "translate-x-5" : "translate-x-0.5"
                  }`}
                />
              </button>
            </label>
            <label className="flex items-center gap-2 text-sm">
              <span className="text-slate-400">Residuals</span>
              <button
                onClick={() => setShowResidual(!showResidual)}
                className={`w-10 h-5 rounded-full transition-colors ${
                  showResidual ? "bg-orange-600" : "bg-slate-700"
                }`}
              >
                <div
                  className={`w-4 h-4 bg-white rounded-full transition-transform ${
                    showResidual ? "translate-x-5" : "translate-x-0.5"
                  }`}
                />
              </button>
            </label>
          </div>
        </div>

        <div className="relative bg-slate-950 rounded-lg p-8 min-h-[500px]">
          {/* Main flow */}
          <div className="flex flex-col items-center gap-4">
            {/* Input */}
            <Block color="#3b82f6" label="Input x" active={step === 0} />
            <Arrow />

            {preNorm && (
              <>
                <Block color="#8b5cf6" label="LayerNorm" active={step === 1} />
                <Arrow />
              </>
            )}

            {/* Attention with residual */}
            <div className="relative">
              <Block color="#ec4899" label="Multi-Head Attention" active={step === 2} />
              {showResidual && (
                <div className="absolute -right-24 top-1/2 -translate-y-1/2 flex items-center gap-2">
                  <div className="w-16 h-0.5 bg-orange-500" />
                  <div className="text-orange-400 text-xs">+ x</div>
                </div>
              )}
            </div>
            <Arrow />

            {!preNorm && (
              <>
                <Block color="#8b5cf6" label="LayerNorm" active={step === 3} />
                <Arrow />
              </>
            )}

            {showResidual && (
              <>
                <Block color="#f97316" label="Add" small active={step === 3} />
                <Arrow />
              </>
            )}

            {preNorm && (
              <>
                <Block color="#8b5cf6" label="LayerNorm" active={step === 4} />
                <Arrow />
              </>
            )}

            {/* FFN with residual */}
            <div className="relative">
              <Block color="#22c55e" label="FFN" active={step === 5} />
              {showResidual && (
                <div className="absolute -right-24 top-1/2 -translate-y-1/2 flex items-center gap-2">
                  <div className="w-16 h-0.5 bg-orange-500" />
                  <div className="text-orange-400 text-xs">+ x</div>
                </div>
              )}
            </div>
            <Arrow />

            {!preNorm && (
              <>
                <Block color="#8b5cf6" label="LayerNorm" active={step === 6} />
                <Arrow />
              </>
            )}

            {showResidual && (
              <>
                <Block color="#f97316" label="Add" small active={step === 6} />
                <Arrow />
              </>
            )}

            {/* Output */}
            <Block color="#06b6d4" label="Output" active={step === 7} />
          </div>

          {/* Residual skip connections visualization */}
          {showResidual && (
            <>
              <svg
                className="absolute inset-0 pointer-events-none"
                style={{ width: "100%", height: "100%" }}
              >
                <path
                  d="M 120 60 C 40 60, 40 180, 120 180"
                  fill="none"
                  stroke="#f97316"
                  strokeWidth="2"
                  strokeDasharray="4"
                  opacity="0.5"
                />
                <path
                  d="M 120 220 C 40 220, 40 340, 120 340"
                  fill="none"
                  stroke="#f97316"
                  strokeWidth="2"
                  strokeDasharray="4"
                  opacity="0.5"
                />
              </svg>
            </>
          )}
        </div>

        {/* Step controls */}
        <div className="mt-4 flex justify-center gap-2">
          {steps.map((s, i) => (
            <button
              key={i}
              onClick={() => setStep(i)}
              className={`w-3 h-3 rounded-full transition-all ${
                step === i ? "scale-125" : "opacity-50"
              }`}
              style={{ backgroundColor: s.color }}
              title={s.name}
            />
          ))}
        </div>
      </div>

      {/* Why residuals */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
          <h2 className="font-semibold text-white mb-4">Why Residual Connections?</h2>
          <div className="space-y-4 text-sm">
            <div>
              <h3 className="text-orange-400 font-medium">Gradient Highway</h3>
              <p className="text-slate-400">
                During backprop, gradients flow through the "+" unmodified. Even if
                sublayer gradient vanishes, original gradient survives.
              </p>
            </div>
            <div>
              <h3 className="text-orange-400 font-medium">Identity Initialization</h3>
              <p className="text-slate-400">
                If F(x)=0 at init, output=x (identity). Easy starting point - model
                learns to ADD refinements.
              </p>
            </div>
            <div>
              <h3 className="text-orange-400 font-medium">Information Preservation</h3>
              <p className="text-slate-400">
                Original input always preserved. "Here's new context, but don't forget
                the original word."
              </p>
            </div>
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
          <h2 className="font-semibold text-white mb-4">Why Layer Normalization?</h2>
          <div className="space-y-4 text-sm">
            <div>
              <h3 className="text-purple-400 font-medium">Stable Activations</h3>
              <p className="text-slate-400">
                Keeps numbers bounded (mean≈0, std≈1) so math doesn't explode across
                deep layers.
              </p>
            </div>
            <div>
              <h3 className="text-purple-400 font-medium">Per-Token Normalization</h3>
              <p className="text-slate-400">
                Unlike BatchNorm, normalizes across features for each token
                independently.
              </p>
            </div>
            <div className="bg-slate-800 rounded p-3 font-mono text-xs">
              <div className="text-purple-400">LayerNorm(x) = γ × (x - μ) / σ + β</div>
              <div className="text-slate-500 mt-1">γ, β are learned parameters</div>
            </div>
          </div>
        </div>
      </div>

      {/* Pre-LN vs Post-LN */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <h2 className="font-semibold text-white mb-4">Pre-LN vs Post-LN</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div className={`p-4 rounded-lg border ${preNorm ? "border-blue-500 bg-blue-900/20" : "border-slate-700"}`}>
            <h3 className="text-white font-medium mb-2">Pre-LN (Modern)</h3>
            <code className="text-xs text-green-400 block mb-2">
              x = x + Attention(LayerNorm(x))
            </code>
            <ul className="text-sm text-slate-400 space-y-1">
              <li>✓ More stable gradients</li>
              <li>✓ No learning rate warmup needed</li>
              <li>✓ Can train deeper models</li>
            </ul>
          </div>
          <div className={`p-4 rounded-lg border ${!preNorm ? "border-blue-500 bg-blue-900/20" : "border-slate-700"}`}>
            <h3 className="text-white font-medium mb-2">Post-LN (Original)</h3>
            <code className="text-xs text-yellow-400 block mb-2">
              x = LayerNorm(x + Attention(x))
            </code>
            <ul className="text-sm text-slate-400 space-y-1">
              <li>• Used in original paper</li>
              <li>• Requires warmup</li>
              <li>• Can have gradient issues</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Stacking */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <h2 className="font-semibold text-white mb-4">Stacking Blocks</h2>
        <p className="text-sm text-slate-400 mb-4">
          This block is repeated N times. Each layer refines the representation further.
        </p>
        <div className="flex items-center justify-center gap-2">
          {Array.from({ length: 6 }).map((_, i) => (
            <div
              key={i}
              className="w-12 h-20 rounded border border-slate-700 bg-gradient-to-b from-pink-900/30 to-green-900/30 flex items-center justify-center text-xs text-slate-400"
            >
              L{i + 1}
            </div>
          ))}
          <div className="text-slate-500 px-2">...</div>
          <div className="text-slate-400 text-sm">
            <div>GPT-2: 12 layers</div>
            <div>GPT-3: 96 layers</div>
          </div>
        </div>
      </div>

      {/* Key insight */}
      <div className="bg-gradient-to-r from-orange-900/30 to-purple-900/30 border border-orange-800/50 rounded-xl p-6">
        <h2 className="font-semibold text-white mb-2">Key Insight</h2>
        <p className="text-slate-300">
          Residual connections are why we can stack 96+ layers. Without them, gradients
          vanish and deep networks can't learn. With them, each layer adds incremental
          refinements while preserving information from earlier layers.
        </p>
      </div>
    </div>
  )
}

function Block({
  color,
  label,
  small = false,
  active = false,
}: {
  color: string
  label: string
  small?: boolean
  active?: boolean
}) {
  return (
    <div
      className={`rounded-lg flex items-center justify-center text-white text-sm font-medium transition-all ${
        active ? "ring-2 ring-white ring-offset-2 ring-offset-slate-950" : ""
      }`}
      style={{
        backgroundColor: color,
        width: small ? 80 : 200,
        height: small ? 32 : 48,
        opacity: active ? 1 : 0.8,
      }}
    >
      {label}
    </div>
  )
}

function Arrow() {
  return (
    <div className="text-slate-600 text-lg">↓</div>
  )
}
