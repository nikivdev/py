import { createOpenAI } from "@ai-sdk/openai"

// Get API key from Cloudflare env or process.env
const getApiKey = (): string | undefined => {
  // Try Cloudflare Workers context first
  try {
    const { getServerContext } = require("@tanstack/react-start/server")
    const ctx = getServerContext()
    if (ctx?.cloudflare?.env?.OPENAI_API_KEY) {
      return ctx.cloudflare.env.OPENAI_API_KEY as string
    }
  } catch {
    // Not in Cloudflare context
  }
  return process.env.OPENAI_API_KEY
}

const getModel = (): string => {
  try {
    const { getServerContext } = require("@tanstack/react-start/server")
    const ctx = getServerContext()
    if (ctx?.cloudflare?.env?.OPENAI_MODEL) {
      return ctx.cloudflare.env.OPENAI_MODEL as string
    }
  } catch {
    // Not in Cloudflare context
  }
  return process.env.OPENAI_MODEL ?? "gpt-4o-mini"
}

export const getOpenAI = () => {
  const apiKey = getApiKey()
  if (!apiKey) {
    return null
  }
  return createOpenAI({ apiKey })
}

export const getDefaultModel = () => getModel()
