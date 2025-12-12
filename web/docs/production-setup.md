# Production Setup Guide

This guide covers deploying the application to Cloudflare Workers with Supabase PostgreSQL and Electric Cloud.

## Prerequisites

- Cloudflare account with Workers enabled
- Supabase project with PostgreSQL database
- Electric Cloud account (or self-hosted Electric instance)
- OpenAI API key (optional, for AI chat)

## 1. Supabase Database Setup

### Enable Logical Replication

Supabase has `wal_level=logical` enabled by default, which is required for ElectricSQL.

### Enable Replication for Tables

Go to **Database → Replication** in Supabase dashboard and enable replication for:
- `users`
- `sessions`
- `accounts`
- `verifications`
- `chat_threads`
- `chat_messages`

### Get Connection String

Copy your connection string from **Settings → Database → Connection string → URI**.

Format: `postgresql://postgres:[PASSWORD]@[PROJECT].supabase.co:5432/postgres`

## 2. Electric Cloud Setup

1. Sign up at [Electric Cloud](https://electric-sql.com/product/cloud)
2. Create a new source connected to your Supabase database
3. Note down:
   - `ELECTRIC_URL` - Your Electric Cloud endpoint
   - `ELECTRIC_SOURCE_ID` - Source identifier
   - `ELECTRIC_SOURCE_SECRET` - Source secret

## 3. Cloudflare Workers Configuration

### Set Secrets

```bash
cd packages/web

# Database
wrangler secret put DATABASE_URL
# Enter: postgresql://postgres:[PASSWORD]@[PROJECT].supabase.co:5432/postgres

# Authentication
wrangler secret put BETTER_AUTH_SECRET
# Enter: A strong random string (at least 32 characters)
# Generate with: openssl rand -hex 32

# Electric Cloud
wrangler secret put ELECTRIC_URL
# Enter: https://your-electric-cloud-url.electric-sql.com

wrangler secret put ELECTRIC_SOURCE_ID
# Enter: Your Electric source ID

wrangler secret put ELECTRIC_SOURCE_SECRET
# Enter: Your Electric source secret

# OpenAI (optional)
wrangler secret put OPENAI_API_KEY
# Enter: sk-...
```

### Set Environment Variables

```bash
# Your production domain
wrangler vars set APP_BASE_URL https://your-app.your-domain.com

# Optional: OpenAI model
wrangler vars set OPENAI_MODEL gpt-4o-mini
```

## 4. Deploy

### Deploy Worker (Backend)

```bash
pnpm deploy:worker
```

### Deploy Web (Frontend)

```bash
pnpm deploy:web
```

### Deploy Both

```bash
pnpm deploy
```

## 5. Verify Deployment

1. Visit your deployed URL
2. Test sign up / sign in
3. Create a chat thread
4. Send a message and verify AI response
5. Check Electric sync is working (open in two tabs)

## Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL connection string (Supabase) |
| `BETTER_AUTH_SECRET` | Yes | Secret for auth session signing |
| `ELECTRIC_URL` | Yes | Electric Cloud or self-hosted URL |
| `ELECTRIC_SOURCE_ID` | Yes* | Electric Cloud source ID |
| `ELECTRIC_SOURCE_SECRET` | Yes* | Electric Cloud source secret |
| `APP_BASE_URL` | Yes | Production URL for CORS/cookies |
| `OPENAI_API_KEY` | No | OpenAI API key for chat AI |
| `OPENAI_MODEL` | No | OpenAI model (default: gpt-4o-mini) |

*Required for Electric Cloud, not needed for self-hosted Electric without auth.

## Troubleshooting

### Auth Issues
- Ensure `APP_BASE_URL` matches your actual domain exactly
- Check `BETTER_AUTH_SECRET` is set and consistent

### Database Connection Issues
- Verify Supabase connection string is correct
- Check if your IP is allowed (Supabase → Settings → Database → Connection Pooling)

### Electric Sync Issues
- Verify tables have replication enabled in Supabase
- Check Electric Cloud dashboard for sync status
- Ensure `ELECTRIC_SOURCE_ID` and `ELECTRIC_SOURCE_SECRET` are correct

### AI Chat Not Working
- Verify `OPENAI_API_KEY` is set and valid
- Check Cloudflare Workers logs for errors
