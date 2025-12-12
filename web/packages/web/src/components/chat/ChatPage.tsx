import { useEffect, useMemo, useState } from "react"
import { useLiveQuery, eq } from "@tanstack/react-db"
import { authClient } from "@/lib/auth-client"
import {
  chatThreadsCollection,
  chatMessagesCollection,
} from "@/lib/collections"

async function createThread(title = "New chat") {
  const res = await fetch("/api/chat/mutations", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ action: "createThread", title }),
  })
  if (!res.ok) throw new Error("Failed to create chat")
  const json = (await res.json()) as { thread: { id: number; title: string } }
  const thread = {
    ...json.thread,
    created_at: json.thread["created_at"]
      ? new Date(json.thread["created_at"])
      : new Date(),
  }
  chatThreadsCollection.insert(thread as any)
  return thread
}

async function addMessage({
  threadId,
  role,
  content,
}: {
  threadId: number
  role: "user" | "assistant"
  content: string
}) {
  const res = await fetch("/api/chat/mutations", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      action: "addMessage",
      threadId,
      role,
      content,
    }),
  })
  if (!res.ok) throw new Error("Failed to add message")
  const json = (await res.json()) as { message: { id: number } & Message }
  const message = {
    ...json.message,
    created_at: json.message.created_at
      ? new Date(json.message.created_at)
      : new Date(),
  }
  chatMessagesCollection.insert(message as any)
  return message
}

async function requestAI({
  threadId,
  prompt,
}: {
  threadId: number
  prompt: string
}) {
  const res = await fetch("/api/chat/ai", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ threadId, prompt }),
  })
  if (!res.ok) throw new Error("AI request failed")
  const json = (await res.json()) as { message: Message }
  const message = {
    ...json.message,
    created_at: json.message.created_at
      ? new Date(json.message.created_at)
      : new Date(),
  }
  chatMessagesCollection.insert(message as any)
  return message
}

type Message = {
  id: number
  thread_id: number
  role: string
  content: string
  created_at: Date
}

export function ChatPage() {
  const { data: session } = authClient.useSession()
  const [input, setInput] = useState("")
  const [activeThreadId, setActiveThreadId] = useState<number | null>(null)
  const [sending, setSending] = useState(false)
  const { data: threads = [] } = useLiveQuery((q) =>
    q
      .from({ chatThreads: chatThreadsCollection })
      .orderBy(({ chatThreads }) => chatThreads.created_at),
  )

  const sortedThreads = useMemo(
    () => [...threads].sort((a, b) => b.id - a.id),
    [threads],
  )

  useEffect(() => {
    if (activeThreadId === null && sortedThreads.length > 0) {
      setActiveThreadId(sortedThreads[0].id)
    }
  }, [sortedThreads, activeThreadId])

  const { data: messages = [] } = useLiveQuery(
    (q) =>
      q
        .from({ chatMessages: chatMessagesCollection })
        .where(({ chatMessages }) =>
          activeThreadId ? eq(chatMessages.thread_id, activeThreadId) : true,
        )
        .orderBy(({ chatMessages }) => chatMessages.created_at),
    [activeThreadId],
  )

  const handleSend = async () => {
    if (!input.trim() || sending) return
    setSending(true)
    try {
      let threadId = activeThreadId
      if (!threadId) {
        const thread = await createThread(input.slice(0, 40) || "New chat")
        threadId = thread.id
        setActiveThreadId(thread.id)
      }
      await addMessage({ threadId, role: "user", content: input })
      await requestAI({ threadId, prompt: input })
      setInput("")
    } catch (error) {
      console.error(error)
      alert("Message failed. Check console.")
    } finally {
      setSending(false)
    }
  }

  if (!session?.session) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p>Please sign in to chat.</p>
      </div>
    )
  }

  return (
    <div className="min-h-screen grid grid-cols-1 md:grid-cols-[280px_1fr]">
      <aside className="border-r border-slate-200 bg-white">
        <div className="p-4 flex justify-between items-center">
          <h2 className="font-semibold text-slate-800">Chats</h2>
          <button
            className="text-sm px-2 py-1 rounded bg-slate-900 text-white"
            onClick={async () => {
              const thread = await createThread()
              setActiveThreadId(thread.id)
            }}
          >
            New
          </button>
        </div>
        <div className="divide-y divide-slate-200">
          {sortedThreads.map((thread) => (
            <button
              key={thread.id}
              className={`w-full text-left px-4 py-3 hover:bg-slate-100 ${
                activeThreadId === thread.id ? "bg-slate-100" : ""
              }`}
              onClick={() => setActiveThreadId(thread.id)}
            >
              <div className="text-sm font-medium text-slate-800">
                {thread.title}
              </div>
              <div className="text-xs text-slate-500">
                {new Date(thread.created_at).toLocaleString()}
              </div>
            </button>
          ))}
          {sortedThreads.length === 0 && (
            <div className="px-4 py-3 text-sm text-slate-500">
              No chats yet. Create one to start talking to the AI.
            </div>
          )}
        </div>
      </aside>
      <main className="flex flex-col bg-slate-50">
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`max-w-2xl rounded-lg px-4 py-3 ${
                msg.role === "assistant"
                  ? "bg-white border border-slate-200"
                  : "bg-slate-900 text-white ml-auto"
              }`}
            >
              <div className="text-xs uppercase tracking-wide mb-1 text-slate-500">
                {msg.role}
              </div>
              <div className="whitespace-pre-wrap text-sm">{msg.content}</div>
            </div>
          ))}
          {messages.length === 0 && (
            <div className="text-slate-500 text-sm">No messages yet.</div>
          )}
        </div>
        <div className="border-t border-slate-200 bg-white p-4">
          <div className="flex gap-3">
            <textarea
              className="flex-1 border border-slate-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-slate-900"
              rows={3}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask the AI anything..."
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault()
                  handleSend()
                }
              }}
            />
            <button
              className="self-end px-4 py-2 bg-slate-900 text-white rounded-lg disabled:opacity-50"
              onClick={handleSend}
              disabled={sending}
            >
              {sending ? "Sending..." : "Send"}
            </button>
          </div>
        </div>
      </main>
    </div>
  )
}
