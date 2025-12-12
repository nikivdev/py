import { createFileRoute } from "@tanstack/react-router"
import { getAuth } from "@/lib/auth"
import { getDb } from "@/db/connection"
import { chat_threads, chat_messages } from "@/db/schema"
import { eq } from "drizzle-orm"

export const Route = createFileRoute("/api/chat/mutations")({
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

        const db = getDb(process.env.DATABASE_URL!)
        const body = await request.json().catch(() => ({}))
        const { action } = body as { action?: string }

        try {
          switch (action) {
            case "createThread": {
              const title =
                (typeof body.title === "string" && body.title.trim()) ||
                "New chat"
              const [thread] = await db
                .insert(chat_threads)
                .values({
                  title,
                  user_id: session.user.id,
                })
                .returning()

              return new Response(
                JSON.stringify({ thread }),
                defaultJsonHeaders(200),
              )
            }
            case "addMessage": {
              const threadId = Number(body.threadId)
              const role =
                typeof body.role === "string" ? body.role.trim() : "user"
              const content =
                typeof body.content === "string" ? body.content.trim() : ""

              if (!threadId || !content || !role) {
                return new Response(
                  JSON.stringify({ error: "Missing threadId/content/role" }),
                  defaultJsonHeaders(400),
                )
              }

              const owner = await db.query.chat_threads.findFirst({
                where(fields, { eq }) {
                  return eq(fields.id, threadId)
                },
              })

              if (!owner || owner.user_id !== session.user.id) {
                return new Response(
                  JSON.stringify({ error: "Forbidden" }),
                  defaultJsonHeaders(403),
                )
              }

              const [message] = await db
                .insert(chat_messages)
                .values({
                  thread_id: threadId,
                  role,
                  content,
                })
                .returning()

              return new Response(
                JSON.stringify({ message }),
                defaultJsonHeaders(200),
              )
            }
            case "renameThread": {
              const threadId = Number(body.threadId)
              const title =
                typeof body.title === "string" ? body.title.trim() : ""
              if (!threadId || !title) {
                return new Response(
                  JSON.stringify({ error: "Missing threadId/title" }),
                  defaultJsonHeaders(400),
                )
              }

              const [thread] = await db
                .update(chat_threads)
                .set({ title })
                .where(eq(chat_threads.id, threadId))
                .returning()

              return new Response(
                JSON.stringify({ thread }),
                defaultJsonHeaders(200),
              )
            }
            default:
              return new Response(
                JSON.stringify({ error: "Unknown action" }),
                defaultJsonHeaders(400),
              )
          }
        } catch (error) {
          console.error("[chat/mutations] error", error)
          return new Response(
            JSON.stringify({ error: "Mutation failed" }),
            defaultJsonHeaders(500),
          )
        }
      },
    },
  },
})

const defaultJsonHeaders = (status: number) => ({
  status,
  headers: { "content-type": "application/json" },
})
