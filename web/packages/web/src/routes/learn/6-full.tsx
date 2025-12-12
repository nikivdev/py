import { createFileRoute } from "@tanstack/react-router"
import { useState } from "react"

export const Route = createFileRoute("/learn/6-full")({
  component: FullLesson,
})

function FullLesson() {
  const [architecture, setArchitecture] = useState<"encoder" | "decoder" | "both">("both")

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">
          6. Full Transformer Architecture
        </h1>
        <p className="text-slate-400">
          Encoder vs Decoder - understanding BERT vs GPT and when to use each.
        </p>
      </div>

      {/* Architecture selector */}
      <div className="flex justify-center gap-2">
        {(["encoder", "decoder", "both"] as const).map((arch) => (
          <button
            key={arch}
            onClick={() => setArchitecture(arch)}
            className={`px-4 py-2 rounded-lg capitalize transition-all ${
              architecture === arch
                ? "bg-blue-600 text-white"
                : "bg-slate-800 text-slate-400 hover:bg-slate-700"
            }`}
          >
            {arch === "both" ? "Encoder-Decoder" : arch}
          </button>
        ))}
      </div>

      {/* Architecture diagram */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <div className="flex justify-center gap-8">
          {/* Encoder */}
          {(architecture === "encoder" || architecture === "both") && (
            <div className="space-y-4">
              <h2 className="text-center font-semibold text-blue-400">Encoder</h2>
              <div className="bg-slate-950 rounded-lg p-4 w-64">
                <ArchBlock label="Input Embedding" color="#3b82f6" />
                <div className="text-center text-slate-500 text-xs my-1">+ Positional</div>
                <div className="border border-dashed border-slate-700 rounded p-3 my-2">
                  <div className="text-xs text-slate-500 mb-2">×N layers</div>
                  <ArchBlock label="Multi-Head Attention" color="#ec4899" />
                  <div className="text-center text-xs text-slate-500 my-1">
                    (Bidirectional)
                  </div>
                  <ArchBlock label="Add & Norm" color="#f97316" small />
                  <ArchBlock label="FFN" color="#22c55e" />
                  <ArchBlock label="Add & Norm" color="#f97316" small />
                </div>
                <ArchBlock label="Encoder Output" color="#06b6d4" />
              </div>
              <div className="text-center text-xs text-slate-400">
                <div>Sees ALL tokens</div>
                <div className="text-slate-500">BERT, RoBERTa</div>
              </div>
            </div>
          )}

          {/* Arrow for encoder-decoder */}
          {architecture === "both" && (
            <div className="flex items-center">
              <div className="text-slate-500 text-2xl">→</div>
            </div>
          )}

          {/* Decoder */}
          {(architecture === "decoder" || architecture === "both") && (
            <div className="space-y-4">
              <h2 className="text-center font-semibold text-purple-400">Decoder</h2>
              <div className="bg-slate-950 rounded-lg p-4 w-64">
                <ArchBlock label="Output Embedding" color="#8b5cf6" />
                <div className="text-center text-slate-500 text-xs my-1">+ Positional</div>
                <div className="border border-dashed border-slate-700 rounded p-3 my-2">
                  <div className="text-xs text-slate-500 mb-2">×N layers</div>
                  <ArchBlock label="Masked Self-Attention" color="#ec4899" />
                  <div className="text-center text-xs text-slate-500 my-1">(Causal)</div>
                  <ArchBlock label="Add & Norm" color="#f97316" small />
                  {architecture === "both" && (
                    <>
                      <ArchBlock label="Cross-Attention" color="#a855f7" />
                      <div className="text-center text-xs text-slate-500 my-1">
                        (to encoder)
                      </div>
                      <ArchBlock label="Add & Norm" color="#f97316" small />
                    </>
                  )}
                  <ArchBlock label="FFN" color="#22c55e" />
                  <ArchBlock label="Add & Norm" color="#f97316" small />
                </div>
                <ArchBlock label="Linear + Softmax" color="#06b6d4" />
                <div className="text-center text-xs text-slate-400 mt-2">→ vocab probs</div>
              </div>
              <div className="text-center text-xs text-slate-400">
                <div>Only sees PAST tokens</div>
                <div className="text-slate-500">GPT, LLaMA, Claude</div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Causal masking visualization */}
      {(architecture === "decoder" || architecture === "both") && (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
          <h2 className="font-semibold text-white mb-4">Causal Masking</h2>
          <p className="text-sm text-slate-400 mb-4">
            In the decoder, position i can only attend to positions 0...i (not future
            tokens)
          </p>
          <div className="flex justify-center">
            <div className="inline-block">
              <div className="flex gap-1 mb-1">
                <div className="w-8" />
                {["I", "love", "to", "code", "!"].map((w, i) => (
                  <div key={i} className="w-10 text-xs text-slate-500 text-center">
                    {w}
                  </div>
                ))}
              </div>
              {["I", "love", "to", "code", "!"].map((word, i) => (
                <div key={i} className="flex gap-1">
                  <div className="w-8 text-xs text-slate-500 flex items-center">
                    {word}
                  </div>
                  {[0, 1, 2, 3, 4].map((j) => (
                    <div
                      key={j}
                      className={`w-10 h-10 rounded flex items-center justify-center text-xs ${
                        j <= i
                          ? "bg-green-600/50 text-green-200"
                          : "bg-slate-800 text-slate-600"
                      }`}
                    >
                      {j <= i ? "✓" : "✗"}
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
          <p className="text-xs text-slate-500 text-center mt-4">
            This forces the model to learn to PREDICT the next token
          </p>
        </div>
      )}

      {/* Comparison table */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <h2 className="font-semibold text-white mb-4">Comparison</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700">
                <th className="text-left py-2 text-slate-400 font-medium">Feature</th>
                <th className="text-left py-2 text-blue-400 font-medium">Encoder</th>
                <th className="text-left py-2 text-purple-400 font-medium">Decoder</th>
                <th className="text-left py-2 text-green-400 font-medium">Enc-Dec</th>
              </tr>
            </thead>
            <tbody className="text-slate-300">
              <tr className="border-b border-slate-800">
                <td className="py-2">Attention</td>
                <td>Bidirectional</td>
                <td>Causal (masked)</td>
                <td>Both</td>
              </tr>
              <tr className="border-b border-slate-800">
                <td className="py-2">Task</td>
                <td>Understanding</td>
                <td>Generation</td>
                <td>Seq2Seq</td>
              </tr>
              <tr className="border-b border-slate-800">
                <td className="py-2">Examples</td>
                <td>BERT, RoBERTa</td>
                <td>GPT, LLaMA</td>
                <td>T5, BART</td>
              </tr>
              <tr className="border-b border-slate-800">
                <td className="py-2">Use cases</td>
                <td>Classification, NER</td>
                <td>Chat, completion</td>
                <td>Translation</td>
              </tr>
              <tr>
                <td className="py-2">Training</td>
                <td>Masked LM</td>
                <td>Next token</td>
                <td>Denoising</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Modern trend */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <h2 className="font-semibold text-white mb-4">Modern Trend: Decoder-Only</h2>
        <p className="text-sm text-slate-400 mb-4">
          Most current LLMs (GPT-4, Claude, LLaMA, Gemini) are decoder-only:
        </p>
        <div className="grid md:grid-cols-3 gap-4 text-sm">
          <div className="bg-slate-800 rounded p-3">
            <div className="text-green-400 font-medium mb-1">Simpler</div>
            <p className="text-slate-400">One model architecture, not two</p>
          </div>
          <div className="bg-slate-800 rounded p-3">
            <div className="text-green-400 font-medium mb-1">Flexible</div>
            <p className="text-slate-400">
              Can do both understanding and generation
            </p>
          </div>
          <div className="bg-slate-800 rounded p-3">
            <div className="text-green-400 font-medium mb-1">Scalable</div>
            <p className="text-slate-400">Easier to scale to massive sizes</p>
          </div>
        </div>
      </div>

      {/* Parameter counts */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <h2 className="font-semibold text-white mb-4">Scale</h2>
        <div className="space-y-3">
          {[
            { name: "BERT-base", params: "110M", layers: 12, d: 768 },
            { name: "GPT-2", params: "1.5B", layers: 48, d: 1600 },
            { name: "GPT-3", params: "175B", layers: 96, d: 12288 },
            { name: "LLaMA-2-70B", params: "70B", layers: 80, d: 8192 },
            { name: "GPT-4", params: "~1.8T", layers: "?", d: "?" },
          ].map((model) => (
            <div key={model.name} className="flex items-center gap-4">
              <div className="w-32 text-slate-400">{model.name}</div>
              <div
                className="h-6 bg-gradient-to-r from-blue-600 to-purple-600 rounded flex items-center px-2 text-xs text-white"
                style={{
                  width: `${Math.min(Math.log10(parseFloat(model.params) * 1e6) * 30, 300)}px`,
                }}
              >
                {model.params}
              </div>
              <div className="text-xs text-slate-500">
                {model.layers} layers, d={model.d}
              </div>
            </div>
          ))}
        </div>
        <p className="text-xs text-slate-500 mt-4">
          Same architecture, just scaled up! More layers, bigger dimensions.
        </p>
      </div>

      {/* Key insight */}
      <div className="bg-gradient-to-r from-blue-900/30 to-purple-900/30 border border-blue-800/50 rounded-xl p-6">
        <h2 className="font-semibold text-white mb-2">Key Insight</h2>
        <p className="text-slate-300">
          The encoder sees everything at once (bidirectional) - great for understanding.
          The decoder can only see the past (causal) - forces it to learn prediction.
          Modern LLMs are decoder-only because generation is the killer app, and with
          enough scale, understanding emerges from the prediction objective.
        </p>
      </div>
    </div>
  )
}

function ArchBlock({
  label,
  color,
  small = false,
}: {
  label: string
  color: string
  small?: boolean
}) {
  return (
    <div
      className={`rounded flex items-center justify-center text-white text-xs font-medium my-1 ${
        small ? "h-6" : "h-10"
      }`}
      style={{ backgroundColor: color }}
    >
      {label}
    </div>
  )
}
