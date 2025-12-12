declare namespace Cloudflare {
  interface Env {
    DATABASE_URL: string
    ELECTRIC_URL: string
    ELECTRIC_SOURCE_ID?: string
    ELECTRIC_SOURCE_SECRET?: string
    BETTER_AUTH_SECRET: string
    APP_BASE_URL?: string
  }
}
