import { createFileRoute, Link, Outlet } from "@tanstack/react-router"

const lessons = [
  { id: "1", name: "Embeddings", path: "/learn/1-embeddings" },
  { id: "2", name: "Self-Attention", path: "/learn/2-attention" },
  { id: "3", name: "Multi-Head", path: "/learn/3-multihead" },
  { id: "4", name: "FFN", path: "/learn/4-ffn" },
  { id: "5", name: "Block", path: "/learn/5-block" },
  { id: "6", name: "Full", path: "/learn/6-full" },
  { id: "7", name: "Training", path: "/learn/7-training" },
]

export const Route = createFileRoute("/learn")({
  component: LearnLayout,
})

function LearnLayout() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <nav className="border-b border-slate-800 bg-slate-900/50 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-3">
          <div className="flex items-center gap-6">
            <Link to="/learn" className="font-bold text-lg text-white">
              Transformers
            </Link>
            <div className="flex gap-1">
              {lessons.map((lesson) => (
                <Link
                  key={lesson.id}
                  to={lesson.path}
                  className="px-3 py-1.5 text-sm rounded-md transition-colors text-slate-400 hover:text-white hover:bg-slate-800 [&.active]:bg-blue-600 [&.active]:text-white"
                >
                  {lesson.id}. {lesson.name}
                </Link>
              ))}
            </div>
          </div>
        </div>
      </nav>
      <main className="max-w-7xl mx-auto px-4 py-8">
        <Outlet />
      </main>
    </div>
  )
}
