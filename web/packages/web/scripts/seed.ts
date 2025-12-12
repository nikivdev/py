import "dotenv/config"
import crypto from "node:crypto"
import { sql, eq } from "drizzle-orm"
import { getDb } from "../src/db/connection"
import {
  accounts,
  chat_messages,
  chat_threads,
  sessions,
  users,
  verifications,
} from "../src/db/schema"

const databaseUrl = process.env.DATABASE_URL

if (!databaseUrl) {
  throw new Error("DATABASE_URL is required in packages/web/.env")
}

const db = getDb(databaseUrl)

async function ensureTables() {
  await db.execute(sql`
    CREATE TABLE IF NOT EXISTS "users" (
      "id" text PRIMARY KEY,
      "name" text NOT NULL,
      "email" text NOT NULL UNIQUE,
      "email_verified" boolean NOT NULL DEFAULT false,
      "image" text,
      "created_at" timestamptz NOT NULL DEFAULT now(),
      "updated_at" timestamptz NOT NULL DEFAULT now()
    );
  `)

  await db.execute(sql`
    CREATE TABLE IF NOT EXISTS "sessions" (
      "id" text PRIMARY KEY,
      "expires_at" timestamptz NOT NULL,
      "token" text NOT NULL UNIQUE,
      "created_at" timestamptz NOT NULL,
      "updated_at" timestamptz NOT NULL,
      "ip_address" text,
      "user_agent" text,
      "user_id" text NOT NULL REFERENCES "users"("id") ON DELETE cascade
    );
  `)

  await db.execute(sql`
    CREATE TABLE IF NOT EXISTS "accounts" (
      "id" text PRIMARY KEY,
      "account_id" text NOT NULL,
      "provider_id" text NOT NULL,
      "user_id" text NOT NULL REFERENCES "users"("id") ON DELETE cascade,
      "access_token" text,
      "refresh_token" text,
      "id_token" text,
      "access_token_expires_at" timestamptz,
      "refresh_token_expires_at" timestamptz,
      "scope" text,
      "password" text,
      "created_at" timestamptz NOT NULL,
      "updated_at" timestamptz NOT NULL
    );
  `)

  await db.execute(sql`
    CREATE TABLE IF NOT EXISTS "verifications" (
      "id" text PRIMARY KEY,
      "identifier" text NOT NULL,
      "value" text NOT NULL,
      "expires_at" timestamptz NOT NULL,
      "created_at" timestamptz,
      "updated_at" timestamptz
    );
  `)

  await db.execute(sql`
    CREATE TABLE IF NOT EXISTS "chat_threads" (
      "id" integer PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
      "title" text NOT NULL,
      "user_id" text NOT NULL,
      "created_at" timestamptz NOT NULL DEFAULT now()
    );
  `)

  await db.execute(sql`
    CREATE TABLE IF NOT EXISTS "chat_messages" (
      "id" integer PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
      "thread_id" integer NOT NULL REFERENCES "chat_threads"("id") ON DELETE cascade,
      "role" varchar(32) NOT NULL,
      "content" text NOT NULL,
      "created_at" timestamptz NOT NULL DEFAULT now()
    );
  `)
}

async function seed() {
  await ensureTables()

  const demoUserId = "demo-user"
  const demoEmail = "demo@ai.chat"

  await db
    .insert(users)
    .values({
      id: demoUserId,
      name: "Demo User",
      email: demoEmail,
      email_verified: true,
      image: null,
      created_at: new Date(),
      updated_at: new Date(),
    })
    .onConflictDoNothing({ target: users.id })

  // Clear any orphaned auth rows for the demo user to keep data tidy
  await db.delete(sessions).where(eq(sessions.user_id, demoUserId))
  await db.delete(accounts).where(eq(accounts.user_id, demoUserId))
  await db.delete(verifications).where(eq(verifications.identifier, demoEmail))

  // Find or create a chat thread for the demo user
  const [existingThread] = await db
    .select()
    .from(chat_threads)
    .where(eq(chat_threads.user_id, demoUserId))
    .limit(1)

  const [thread] =
    existingThread && existingThread.id
      ? [existingThread]
      : await db
          .insert(chat_threads)
          .values({
            title: "Getting started with AI chat",
            user_id: demoUserId,
          })
          .returning()

  const threadId = thread.id

  await db.delete(chat_messages).where(eq(chat_messages.thread_id, threadId))

  const starterMessages = [
    {
      role: "user",
      content: "How do I get reliable AI chat responses from this app?",
    },
    {
      role: "assistant",
      content:
        "Each thread keeps your message history. You can seed demos like this one, or stream responses from your AI provider. Try adding more messages to this thread.",
    },
    {
      role: "user",
      content: "Can I hook this up to my own model API?",
    },
    {
      role: "assistant",
      content:
        "Yes. Point your server-side handler at your model endpoint and persist messages into the database. Electric can sync them live to the client.",
    },
  ]

  await db.insert(chat_messages).values(
    starterMessages.map((msg) => ({
      thread_id: threadId,
      role: msg.role,
      content: msg.content,
      created_at: new Date(),
    })),
  )
}

seed()
  .then(() => {
    console.log("Seed complete: demo user and chat thread ready.")
  })
  .catch((err) => {
    console.error(err)
    process.exit(1)
  })
