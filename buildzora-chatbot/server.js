// BuildZora Chatbot Backend
// Keeps the Anthropic API key server-side and proxies chat requests from the widget.

const express = require('express');
const path = require('path');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
const MODEL = process.env.ANTHROPIC_MODEL || 'claude-sonnet-4-6';

if (!ANTHROPIC_API_KEY) {
  console.warn('[WARN] ANTHROPIC_API_KEY is not set. Add it to a .env file before going live.');
}

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// ---- Editable knowledge base: replace with BuildZora's real FAQ/policy content ----
const KNOWLEDGE_BASE = `
Company: BuildZora — supplier of building materials and construction project services.
Products: structural steel, cement & aggregates, timber, roofing sheets, plumbing & electrical fittings, on-site delivery.
Ordering: orders placed via website or by calling the regional depot; minimum order value may apply for free delivery.
Delivery: standard delivery 2-5 business days within city limits; site delivery scheduling available for large orders; delays possible during monsoon season.
Returns: unused, unopened materials can be returned within 14 days with the original invoice; custom-cut or mixed materials are non-returnable.
Payments: accepts UPI, bank transfer, major cards, and approved credit accounts for contractors.
Quotes: bulk/project quotes are prepared by the sales team within 24 hours of a request; include project size, location and material list for a faster quote.
Support hours: Monday-Saturday, 9am-7pm local time. Closed on public holidays.
Warranty: manufacturer warranties apply per product; BuildZora does not warranty labour performed by third-party contractors.
`.trim();

const SYSTEM_PROMPT = `You are the customer support assistant for BuildZora, a building materials and construction services company. Answer only using the knowledge base below. Be concise, warm, and practical — a few sentences at most. If the answer isn't in the knowledge base, say you don't have that detail and suggest contacting the support team (Mon–Sat, 9am–7pm), rather than guessing.

KNOWLEDGE BASE:
${KNOWLEDGE_BASE}`;

// Simple in-memory rate limiter (per IP) — replace with a real one (e.g. express-rate-limit) for production.
const rateMap = new Map();
const RATE_LIMIT = 20; // requests
const RATE_WINDOW_MS = 60 * 1000; // per 1 minute

function isRateLimited(ip) {
  const now = Date.now();
  const entry = rateMap.get(ip) || { count: 0, start: now };
  if (now - entry.start > RATE_WINDOW_MS) {
    entry.count = 0;
    entry.start = now;
  }
  entry.count += 1;
  rateMap.set(ip, entry);
  return entry.count > RATE_LIMIT;
}

app.post('/api/chat', async (req, res) => {
  try {
    const ip = req.headers['x-forwarded-for'] || req.socket.remoteAddress;
    if (isRateLimited(ip)) {
      return res.status(429).json({ error: 'Too many requests. Please slow down.' });
    }

    const { messages } = req.body;
    if (!Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ error: 'messages array is required' });
    }
    if (!ANTHROPIC_API_KEY) {
      return res.status(500).json({ error: 'Server is missing ANTHROPIC_API_KEY.' });
    }

    // Only forward role + content, and cap history length to keep costs sane
    const trimmed = messages.slice(-12).map(m => ({ role: m.role, content: String(m.content).slice(0, 4000) }));

    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify({
        model: MODEL,
        max_tokens: 500,
        system: SYSTEM_PROMPT,
        messages: trimmed
      })
    });

    if (!response.ok) {
      const errText = await response.text();
      console.error('Anthropic API error:', response.status, errText);
      return res.status(502).json({ error: 'Upstream AI request failed.' });
    }

    const data = await response.json();
    const text = (data.content || [])
      .filter(block => block.type === 'text')
      .map(block => block.text)
      .join('\n');

    res.json({ reply: text || "Sorry, I couldn't generate a reply — please try again." });
  } catch (err) {
    console.error('Chat handler error:', err);
    res.status(500).json({ error: 'Something went wrong on our end.' });
  }
});

app.listen(PORT, () => {
  console.log(`BuildZora chatbot backend running at http://localhost:${PORT}`);
});
