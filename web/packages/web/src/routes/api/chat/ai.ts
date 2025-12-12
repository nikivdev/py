import { createFileRoute } from "@tanstack/react-router"
import { streamText } from "ai"
import { getAuth } from "@/lib/auth"
import { getDb } from "@/db/connection"
import { chat_messages, chat_threads } from "@/db/schema"
import { getOpenAI, getDefaultModel } from "@/lib/ai/provider"

export const Route = createFileRoute("/api/chat/ai")({
  server: {
    handlers: {
      POST: async ({ request }) => {
        const session = await getAuth().api.getSession({
          headers: request.headers,
        })
        if (!session?.user?.id) {
          return new Response(JSON.stringify({ error: "Unauthorized" }), {
            status: 401,
            headers: { "content-type": "application/json" },
          })
        }

        const body = (await request.json().catch(() => ({}))) as {
          threadId?: number | string
          messages?: Array<{ role: "user" | "assistant"; content: string }>
        }

        const threadId = Number(body.threadId)
        const messages = body.messages ?? []

        if (!threadId || messages.length === 0) {
          return new Response(
            JSON.stringify({ error: "Missing threadId or messages" }),
            {
              status: 400,
              headers: { "content-type": "application/json" },
            },
          )
        }

        const db = getDb(process.env.DATABASE_URL!)

        // Verify thread ownership
        const thread = await db.query.chat_threads.findFirst({
          where(fields, { eq }) {
            return eq(fields.id, threadId)
          },
        })

        if (!thread || thread.user_id !== session.user.id) {
          return new Response(JSON.stringify({ error: "Forbidden" }), {
            status: 403,
            headers: { "content-type": "application/json" },
          })
        }

        const openai = getOpenAI()
        if (!openai) {
          // Fallback to non-streaming demo response
          const lastUserMessage = messages
            .filter((m) => m.role === "user")
            .pop()
          const reply = `Demo reply: I received "${lastUserMessage?.content}". Configure OPENAI_API_KEY for real responses.`

          // Save the assistant message
          await db.insert(chat_messages).values({
            thread_id: threadId,
            role: "assistant",
            content: reply,
          })

          return new Response(JSON.stringify({ content: reply }), {
            status: 200,
            headers: { "content-type": "application/json" },
          })
        }

        // Use AI SDK streaming
        const result = streamText({
          model: openai(getDefaultModel()),
          system: "You are a helpful assistant.",
          messages: messages.map((m) => ({
            role: m.role,
            content: m.content,
          })),
          async onFinish({ text }) {
            // Save the assistant message when streaming completes
            await db.insert(chat_messages).values({
              thread_id: threadId,
              role: "assistant",
              content: text,
            })
          },
        })

        // Return the streaming response
        return result.toDataStreamResponse()
      },
    },
  },
})
