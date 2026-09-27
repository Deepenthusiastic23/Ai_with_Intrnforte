# BuildZora Chatbot

AI customer support widget for BuildZora. The API key stays server-side (Express backend) — safe to deploy on your own domain.

## 1. Setup

```bash
npm install
cp .env.example .env
```

Open `.env` and paste your real Anthropic API key:
```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxx
```
Get a key from the Claude Console (console.anthropic.com) if you don't have one yet.

## 2. Run locally

```bash
npm start
```

Open http://localhost:3000 — the widget loads and the AI actually replies now, using your API key.

## 3. Customize

Edit the `KNOWLEDGE_BASE` constant in `server.js` with BuildZora's real FAQs, policies, and product info. The assistant only answers from that text, so it won't invent facts.

## 4. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: BuildZora chatbot"
git remote add origin https://github.com/yourname/buildzora-chatbot.git
git branch -M main
git push -u origin main
```

`.gitignore` already excludes `.env` and `node_modules`, so your API key never gets committed.

## 5. Deploy

Any Node-friendly host works: Render, Railway, Fly.io, or a VPS. General steps:

1. Push this repo to GitHub (above).
2. Connect the repo on your hosting platform.
3. Set the environment variable `ANTHROPIC_API_KEY` (and optionally `ANTHROPIC_MODEL`) in the platform's dashboard — never in code.
4. Set the start command to `npm start`.
5. Once deployed, embed the widget on your main site with an iframe, or copy `public/index.html`'s widget markup directly into your site and point its `fetch('/api/chat')` calls at your deployed backend's URL.

## Notes

- A basic in-memory rate limiter is included (20 requests/minute per IP). Swap in `express-rate-limit` + Redis for real production traffic.
- Chat history is kept only in the browser tab (not stored server-side). Add a database if you want conversation logs or analytics.
