import { createFileRoute } from "@tanstack/react-router"
import { getAuth } from "@/lib/auth"
import { prepareElectricUrl, proxyElectricRequest } from "@/lib/electric-proxy"
import { getDb } from "@/db/connection"

const serve = async ({ request }: { request: Request }) => {
  const session = await getAuth().api.getSession({ headers: request.headers })
  if (!session?.user?.id) {
    return new Response(JSON.stringify({ error: "Unauthorized" }), {
      status: 401,
      headers: { "content-type": "application/json" },
    })
  }

  // Ensure tables exist before proxying
  try {
    const db = getDb(process.env.DATABASE_URL!)
    await db.execute(
      `CREATE TABLE IF NOT EXISTS chat_threads (
        id integer PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
        title text NOT NULL,
        user_id text NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now()
      );`,
    )
    await db.execute(
      `CREATE TABLE IF NOT EXISTS chat_messages (
        id integer PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
        thread_id integer NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
        role varchar(32) NOT NULL,
        content text NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now()
      );`,
    )
  } catch (error) {
    console.warn("[chat-messages] ensure table failed", error)
  }

  const originUrl = prepareElectricUrl(request.url)
  originUrl.searchParams.set("table", "chat_messages")

  // Limit messages to threads owned by the user
  originUrl.searchParams.set(
    "where",
    `"thread_id" IN (SELECT id FROM chat_threads WHERE user_id = '${session.user.id}')`,
  )

  return proxyElectricRequest(originUrl)
}

export const Route = createFileRoute("/api/chat-messages")({
  server: {
    handlers: {
      GET: serve,
    },
  },
})
