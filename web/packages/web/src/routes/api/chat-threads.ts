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

  // Ensure the table exists before proxying (helpful for first run)
  try {
    const db = getDb(process.env.DATABASE_URL!)
    await db.execute(
      `CREATE TABLE IF NOT EXISTS chat_threads (
        id integer PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
        title text NOT NULL,
        user_id text NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now()
      )`,
    )
  } catch (error) {
    console.warn("[chat-threads] ensure table failed", error)
  }

  const originUrl = prepareElectricUrl(request.url)
  originUrl.searchParams.set("table", "chat_threads")
  const filter = `"user_id" = '${session.user.id}'`
  originUrl.searchParams.set("where", filter)

  return proxyElectricRequest(originUrl)
}

export const Route = createFileRoute("/api/chat-threads")({
  server: {
    handlers: {
      GET: serve,
    },
  },
})
