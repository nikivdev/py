import { betterAuth } from "better-auth"
import { drizzleAdapter } from "better-auth/adapters/drizzle"
import { tanstackStartCookies } from "better-auth/tanstack-start"
import { emailOTP } from "better-auth/plugins"
import { getAuthDb } from "@/db/connection"
import * as schema from "@/db/schema"

type AuthEnv = {
  DATABASE_URL: string
  BETTER_AUTH_SECRET: string
  APP_BASE_URL?: string
}

let cachedAuth: ReturnType<typeof betterAuth> | null = null
let cachedDbUrl = ""

// Get env from Cloudflare context or process.env
const getEnv = (): AuthEnv => {
  let DATABASE_URL: string | undefined
  let BETTER_AUTH_SECRET: string | undefined
  let APP_BASE_URL: string | undefined

  // Try Cloudflare Workers context first (production)
  try {
    const { getServerContext } = require("@tanstack/react-start/server")
    const ctx = getServerContext()
    if (ctx?.cloudflare?.env) {
      const cfEnv = ctx.cloudflare.env as Partial<AuthEnv>
      DATABASE_URL = cfEnv.DATABASE_URL
      BETTER_AUTH_SECRET = cfEnv.BETTER_AUTH_SECRET
      APP_BASE_URL = cfEnv.APP_BASE_URL
    }
  } catch {
    // Not in Cloudflare context
  }

  // Fall back to process.env (local dev)
  DATABASE_URL = DATABASE_URL ?? process.env.DATABASE_URL
  BETTER_AUTH_SECRET = BETTER_AUTH_SECRET ?? process.env.BETTER_AUTH_SECRET
  APP_BASE_URL = APP_BASE_URL ?? process.env.APP_BASE_URL

  if (!DATABASE_URL) {
    throw new Error("DATABASE_URL is not configured")
  }
  if (!BETTER_AUTH_SECRET) {
    throw new Error("BETTER_AUTH_SECRET is not configured")
  }

  return { DATABASE_URL, BETTER_AUTH_SECRET, APP_BASE_URL }
}

export const getAuth = () => {
  const env = getEnv()

  if (!cachedAuth || cachedDbUrl !== env.DATABASE_URL) {
    const db = getAuthDb(env.DATABASE_URL)
    cachedDbUrl = env.DATABASE_URL

    const isDev = process.env.NODE_ENV !== "production"

    cachedAuth = betterAuth({
      database: drizzleAdapter(db, {
        provider: "pg",
        usePlural: true,
        schema,
      }),
      trustedOrigins: [env.APP_BASE_URL ?? "http://localhost:8712"],
      plugins: [
        tanstackStartCookies(),
        emailOTP({
          async sendVerificationOTP({ email, otp }) {
            if (isDev) {
              // In dev mode, log OTP to terminal
              console.log("\n" + "=".repeat(50))
              console.log(`🔐 OTP CODE for ${email}`)
              console.log(`   Code: ${otp}`)
              console.log("=".repeat(50) + "\n")
            } else {
              // In production, send email via your email provider
              // TODO: Implement email sending (e.g., Resend, SendGrid, etc.)
              console.log(`[EMAIL] Would send OTP ${otp} to ${email}`)
            }
          },
          otpLength: 6,
          expiresIn: 300, // 5 minutes
        }),
      ],
    })
  }

  return cachedAuth
}

// Lazy proxy that calls getAuth() on each access
export const auth = new Proxy({} as ReturnType<typeof betterAuth>, {
  get(_target, prop) {
    return getAuth()[prop as keyof ReturnType<typeof betterAuth>]
  },
})
