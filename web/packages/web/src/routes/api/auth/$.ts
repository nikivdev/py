import { createFileRoute } from "@tanstack/react-router"
import { getAuth } from "@/lib/auth"

const notConfigured = () =>
  new Response(JSON.stringify({ error: "Auth not configured" }), {
    status: 503,
    headers: { "Content-Type": "application/json" },
  })

export const Route = createFileRoute("/api/auth/$")({
  server: {
    handlers: {
      GET: ({ request }) => getAuth()?.handler(request) ?? notConfigured(),
      POST: ({ request }) => getAuth()?.handler(request) ?? notConfigured(),
    },
  },
})
