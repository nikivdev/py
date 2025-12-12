import { createFileRoute, Link } from "@tanstack/react-router"

export const Route = createFileRoute("/learn/")({
  component: LearnIndex,
})

const lessons = [
  {
    id: 1,
    title: "Embeddings + Positional Encoding",
    desc: "How words become vectors, and how position is encoded",
    color: "from-blue-500 to-cyan-500",
  },
  {
    id: 2,
    title: "Self-Attention (Q, K, V)",
    desc: "The heart of transformers - how words attend to each other",
    color: "from-purple-500 to-pink-500",
  },
  {
    id: 3,
    title: "Multi-Head Attention",
    desc: "Running attention in parallel to capture different relationships",
    color: "from-orange-500 to-red-500",
  },
  {
    id: 4,
    title: "Feed-Forward Network",
    desc: "The 'thinking' phase - where knowledge is stored",
    color: "from-green-500 to-emerald-500",
  },
  {
    id: 5,
    title: "Transformer Block",
    desc: "Residual connections and layer normalization",
    color: "from-yellow-500 to-orange-500",
  },
  {
    id: 6,
    title: "Full Architecture",
    desc: "Encoder vs Decoder - BERT vs GPT",
    color: "from-indigo-500 to-purple-500",
  },
  {
    id: 7,
    title: "Training Demo",
    desc: "Watch weights learn through backpropagation",
    color: "from-rose-500 to-pink-500",
  },
]

function LearnIndex() {
  return (
    <div className="space-y-8">
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
          Understanding Transformers
        </h1>
        <p className="text-slate-400 max-w-2xl mx-auto">
          Move beyond "it pays attention to words" - dive into the flow of vectors,
          the math of attention, and why this architecture changed everything.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {lessons.map((lesson) => (
          <Link
            key={lesson.id}
            to={`/learn/${lesson.id}-${lesson.title.split(" ")[0].toLowerCase()}`}
            className="group relative overflow-hidden rounded-xl bg-slate-900 border border-slate-800 p-6 hover:border-slate-700 transition-all hover:scale-[1.02]"
          >
            <div
              className={`absolute inset-0 bg-gradient-to-br ${lesson.color} opacity-5 group-hover:opacity-10 transition-opacity`}
            />
            <div className="relative space-y-2">
              <div className="flex items-center gap-3">
                <span
                  className={`w-8 h-8 rounded-lg bg-gradient-to-br ${lesson.color} flex items-center justify-center text-white font-bold text-sm`}
                >
                  {lesson.id}
                </span>
                <h2 className="font-semibold text-white">{lesson.title}</h2>
              </div>
              <p className="text-sm text-slate-400">{lesson.desc}</p>
            </div>
          </Link>
        ))}
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <h2 className="font-semibold text-white mb-4">The Core Philosophy</h2>
        <div className="grid md:grid-cols-2 gap-6 text-sm">
          <div className="space-y-2">
            <h3 className="text-red-400 font-medium">Before (RNNs)</h3>
            <p className="text-slate-400">
              Read word 1 → store context → Read word 2 → update context...
            </p>
            <p className="text-slate-500">
              Sequential. Slow. Forgets long sequences.
            </p>
          </div>
          <div className="space-y-2">
            <h3 className="text-green-400 font-medium">Transformers</h3>
            <p className="text-slate-400">
              Read ALL words at once. One massive matrix operation.
            </p>
            <p className="text-slate-500">
              Parallel. Fast. Handles long context.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
