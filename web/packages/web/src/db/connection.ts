import { neon, neonConfig } from "@neondatabase/serverless"
import { drizzle as drizzleNeon } from "drizzle-orm/neon-http"
import * as schema from "./schema"

const dbCache = new Map<string, unknown>()
const authDbCache = new Map<string, unknown>()

// Configure Neon for local development via local-neon-http-proxy
// The proxy runs at localhost:4444 and translates HTTP to postgres protocol
const configureNeonForLocal = () => {
  // Point to local neon-http-proxy instead of Neon cloud
  neonConfig.fetchEndpoint = (host) => {
    const protocol = host === "db.localtest.me" ? "http" : "https"
    const port = host === "db.localtest.me" ? 4444 : 443
    return `${protocol}://${host}:${port}/sql`
  }
  // Use local proxy for WebSocket connections too
  neonConfig.wsProxy = (host) => `${host}:4444/v2`
  neonConfig.useSecureWebSocket = false
  neonConfig.pipelineTLS = false
  neonConfig.pipelineConnect = false
}

// Check if we're using local postgres (via neon proxy)
const isLocalDbUrl = (url: string) => {
  try {
    const { hostname } = new URL(url)
    return (
      hostname === "localhost" ||
      hostname === "127.0.0.1" ||
      hostname === "db.localtest.me"
    )
  } catch {
    return false
  }
}

// Main db with snake_case casing for app tables (chat_threads, chat_messages)
export const getDb = (databaseUrl: string) => {
  if (dbCache.has(databaseUrl)) {
    return dbCache.get(databaseUrl)!
  }

  if (isLocalDbUrl(databaseUrl)) {
    configureNeonForLocal()
  } else {
    neonConfig.fetchConnectionCache = true
  }

  const sql = neon(databaseUrl)
  const db = drizzleNeon(sql, { schema, casing: "snake_case" })

  dbCache.set(databaseUrl, db)
  return db
}

// Auth db WITHOUT casing transform for better-auth tables (users, sessions, etc.)
// better-auth uses camelCase columns and manages its own naming
export const getAuthDb = (databaseUrl: string) => {
  if (authDbCache.has(databaseUrl)) {
    return authDbCache.get(databaseUrl)!
  }

  if (isLocalDbUrl(databaseUrl)) {
    configureNeonForLocal()
  } else {
    neonConfig.fetchConnectionCache = true
  }

  const sql = neon(databaseUrl)
  const db = drizzleNeon(sql, { schema })

  authDbCache.set(databaseUrl, db)
  return db
}
