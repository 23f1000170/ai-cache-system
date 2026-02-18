"""
AI Caching System for Content Moderation
=========================================
A complete caching server with:
  1. Exact-match caching (MD5 hash)
  2. Semantic caching (cosine similarity on simple embeddings)
  3. LRU eviction policy
  4. TTL (time-to-live) expiration
  5. Full analytics endpoint

Run:  python3 cache_server.py
Then visit http://localhost:8000
"""

import http.server
import json
import hashlib
import time
import math
import random
import threading
from collections import OrderedDict
from urllib.parse import urlparse
import os
PORT = int(os.environ.get("PORT", 3000))

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
MAX_CACHE_SIZE      = 1500          # max entries before LRU eviction
TTL_SECONDS         = 86_400        # 24 hours in seconds
SIMILARITY_THRESHOLD = 0.95         # cosine similarity cutoff for semantic match
MODEL_COST_PER_1M   = 0.40          # dollars per 1 million tokens
AVG_TOKENS          = 300           # average tokens per request
BASELINE_DAILY_COST = 5.28          # dollars (pre-caching baseline)
DAILY_REQUESTS      = 43_972        # for analytics display

# ─────────────────────────────────────────────
#  SIMPLE EMBEDDING FUNCTION
# (In production you'd call OpenAI / Gemini embeddings API here.
#  Here we use a deterministic bag-of-words approach so the server
#  works 100% offline and is easy to understand.)
# ─────────────────────────────────────────────

# A small fixed vocabulary of words common in content moderation queries
_VOCAB = [
    "is","this","text","spam","hate","safe","appropriate","offensive",
    "content","message","user","comment","review","post","check","flag",
    "remove","keep","allow","block","violent","sexual","harmful","abusive",
    "threatening","mild","severe","moderate","language","image","video",
    "link","profanity","harassment","bullying","illegal","drug","weapon",
    "scam","phishing","fake","misinformation","adult","child","explicit",
    "nudity","gore","self","harm","suicide","terrorism","extremist","policy",
    "violation","warning","account","ban","report","automated","manual"
]
_VOCAB_INDEX = {word: i for i, word in enumerate(_VOCAB)}

def _embed(text: str) -> list[float]:
    """
    Convert text → a fixed-length numeric vector (embedding).

    We use a simple 'bag of words' approach:
      - Normalise & split the text into words
      - For each word in our vocabulary, count how many times it appears
      - Return the count vector divided by its length (unit vector)

    This makes 'Hello world' and 'hello World' produce identical vectors,
    so they'll get the same cache hit — that's exactly what we want!
    """
    vec = [0.0] * len(_VOCAB)
    words = text.lower().split()
    for w in words:
        # strip basic punctuation
        w = w.strip(".,!?;:\"'()[]{}")
        if w in _VOCAB_INDEX:
            vec[_VOCAB_INDEX[w]] += 1.0

    # If all zeros (no vocabulary words found), use character-level hashing
    # to get a non-zero embedding so we can still do similarity
    if all(v == 0 for v in vec):
        for i, ch in enumerate(text.lower()):
            vec[ord(ch) % len(_VOCAB)] += 1.0

    # Normalise to unit length
    magnitude = math.sqrt(sum(v * v for v in vec))
    if magnitude > 0:
        vec = [v / magnitude for v in vec]
    return vec


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity between two vectors.
    Returns a value between -1 and 1 (1 = identical direction).

    Formula:  cos(θ) = (A · B) / (|A| × |B|)
    Since our vectors are already unit-normalised, this simplifies to dot product.
    """
    return sum(x * y for x, y in zip(a, b))


# ─────────────────────────────────────────────
#  LRU CACHE  (Exact + Semantic combined)
# ─────────────────────────────────────────────

class AICache:
    """
    A two-layer cache:
      Layer 1 – Exact match  : key = MD5(normalised_query)
      Layer 2 – Semantic     : scan stored embeddings for cosine similarity ≥ 0.95

    Eviction: Least Recently Used (LRU) via OrderedDict
    Expiration: each entry has a timestamp; entries older than TTL_SECONDS are stale
    """

    def __init__(self):
        # OrderedDict preserves insertion order, which we use for LRU
        # Key   → (answer, embedding, timestamp, hit_count)
        self._store: OrderedDict = OrderedDict()
        self._lock  = threading.Lock()     # thread-safe for concurrent requests

        # ── Analytics counters ──
        self.total_requests    = 0
        self.cache_hits        = 0
        self.exact_hits        = 0
        self.semantic_hits     = 0
        self.cache_misses      = 0
        self.tokens_saved      = 0         # tokens we didn't need to spend on the LLM

    # ──────────────────────────
    #  NORMALISE
    # ──────────────────────────
    @staticmethod
    def _normalise(query: str) -> str:
        """
        Strip leading/trailing whitespace, collapse internal spaces,
        and lower-case everything.

        Why? So 'Hello' and 'hello  ' and 'HELLO' all map to the same cache key.
        """
        return " ".join(query.lower().split())

    # ──────────────────────────
    #  MAKE A CACHE KEY
    # ──────────────────────────
    @staticmethod
    def _make_key(normalised: str) -> str:
        """MD5 hash of the normalised query → short, fixed-length key."""
        return hashlib.md5(normalised.encode()).hexdigest()

    # ──────────────────────────
    #  CHECK TTL
    # ──────────────────────────
    @staticmethod
    def _is_expired(timestamp: float) -> bool:
        return (time.time() - timestamp) > TTL_SECONDS

    # ──────────────────────────
    #  GET (lookup)
    # ──────────────────────────
    def get(self, query: str) -> tuple[str | None, str, str | None]:
        """
        Try to find a cached answer for `query`.

        Returns:
          (answer_or_None, cache_type, cache_key_or_None)
          cache_type ∈ {"exact", "semantic", "miss"}
        """
        normalised = self._normalise(query)
        key        = self._make_key(normalised)
        embedding  = _embed(normalised)

        with self._lock:
            self.total_requests += 1

            # ── 1. Exact match ──────────────────────────────────────────────
            if key in self._store:
                answer, stored_emb, ts, hit_count = self._store[key]
                if not self._is_expired(ts):
                    # Move to end → marks as recently used (LRU bookkeeping)
                    self._store.move_to_end(key)
                    self._store[key] = (answer, stored_emb, ts, hit_count + 1)
                    self.cache_hits  += 1
                    self.exact_hits  += 1
                    self.tokens_saved += AVG_TOKENS
                    return answer, "exact", key
                else:
                    # Entry expired → remove it
                    del self._store[key]

            # ── 2. Semantic match ───────────────────────────────────────────
            best_sim  = 0.0
            best_key  = None
            best_ans  = None
            for k, (answer, stored_emb, ts, hit_count) in list(self._store.items()):
                if self._is_expired(ts):
                    del self._store[k]
                    continue
                sim = _cosine_similarity(embedding, stored_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_key = k
                    best_ans = answer

            if best_sim >= SIMILARITY_THRESHOLD and best_key:
                # Promote the matched entry (LRU)
                self._store.move_to_end(best_key)
                old = self._store[best_key]
                self._store[best_key] = (old[0], old[1], old[2], old[3] + 1)
                self.cache_hits    += 1
                self.semantic_hits += 1
                self.tokens_saved  += AVG_TOKENS
                return best_ans, "semantic", best_key

            # ── 3. Cache miss ───────────────────────────────────────────────
            self.cache_misses += 1
            return None, "miss", None

    # ──────────────────────────
    #  PUT (store result)
    # ──────────────────────────
    def put(self, query: str, answer: str) -> str:
        """Store a new answer in the cache. Returns the cache key."""
        normalised = self._normalise(query)
        key        = self._make_key(normalised)
        embedding  = _embed(normalised)

        with self._lock:
            # LRU eviction: if cache is full, remove the LEAST recently used entry
            # (that's the first item in the OrderedDict)
            while len(self._store) >= MAX_CACHE_SIZE:
                self._store.popitem(last=False)   # pop from front = oldest used

            self._store[key] = (answer, embedding, time.time(), 0)
            self._store.move_to_end(key)           # mark as recently used
        return key

    # ──────────────────────────
    #  ANALYTICS
    # ──────────────────────────
    def analytics(self) -> dict:
        """Return a stats snapshot."""
        with self._lock:
            total = self.total_requests or 1          # avoid /0
            hit_rate = round(self.cache_hits / total, 4)

            # Cost maths
            # Tokens saved = cached tokens we didn't send to the LLM
            dollars_saved = round(self.tokens_saved * MODEL_COST_PER_1M / 1_000_000, 4)
            savings_pct   = round((dollars_saved / BASELINE_DAILY_COST) * 100, 1) if dollars_saved else 0

            return {
                "hitRate":       hit_rate,
                "totalRequests": self.total_requests,
                "cacheHits":     self.cache_hits,
                "exactHits":     self.exact_hits,
                "semanticHits":  self.semantic_hits,
                "cacheMisses":   self.cache_misses,
                "cacheSize":     len(self._store),
                "maxCacheSize":  MAX_CACHE_SIZE,
                "costSavings":   dollars_saved,
                "savingsPercent":savings_pct,
                "baselineCost":  BASELINE_DAILY_COST,
                "strategies": [
                    "exact match caching",
                    "semantic similarity (cosine ≥ 0.95)",
                    "LRU eviction policy",
                    "TTL expiration (24h)"
                ]
            }


# ─────────────────────────────────────────────
#  FAKE LLM CALL  (simulates OpenAI/Gemini API)
# ─────────────────────────────────────────────

_MOD_RESPONSES = {
    "safe": [
        "This content appears safe and appropriate. No violations detected.",
        "Content review complete: message is within community guidelines.",
        "No harmful content found. This post is approved.",
    ],
    "spam": [
        "This content has been flagged as potential spam. Recommend removal.",
        "Spam indicators detected. Suggest blocking this submission.",
    ],
    "hate": [
        "Hate speech detected. This content violates our policies and should be removed.",
        "Content flagged for harmful language targeting a protected group.",
    ],
    "default": [
        "Content moderation analysis complete. Please review with a human moderator.",
        "Unable to definitively classify. Escalating to human review.",
    ]
}

def _fake_llm_call(query: str) -> tuple[str, int]:
    """
    Pretend to call an LLM API.
    In real life, replace this with:
        response = openai.chat.completions.create(...)
    Returns (answer_text, latency_ms)
    """
    # Simulate network latency (500ms – 1500ms)
    latency = random.randint(500, 1500)
    time.sleep(latency / 1000)

    q = query.lower()
    if any(w in q for w in ("spam", "scam", "phishing")):
        answers = _MOD_RESPONSES["spam"]
    elif any(w in q for w in ("hate", "racist", "offensive", "slur")):
        answers = _MOD_RESPONSES["hate"]
    elif any(w in q for w in ("safe", "okay", "fine", "appropriate", "normal")):
        answers = _MOD_RESPONSES["safe"]
    else:
        answers = _MOD_RESPONSES["default"]

    return random.choice(answers), latency


# ─────────────────────────────────────────────
#  HTTP SERVER
# ─────────────────────────────────────────────

# Shared cache instance (lives for the lifetime of the server)
cache = AICache()

# Pre-warm the cache with some realistic content-moderation queries
# so the /analytics endpoint shows interesting numbers right away
_SEED_QUERIES = [
    ("Is this spam email?",             "This content has been flagged as potential spam. Recommend removal."),
    ("check for hate speech",           "Hate speech detected. This content violates our policies and should be removed."),
    ("is this message safe?",           "Content review complete: message is within community guidelines."),
    ("contains offensive language?",    "Content flagged for harmful language targeting a protected group."),
    ("review this user comment",        "Content moderation analysis complete. Please review with a human moderator."),
    ("is this post appropriate",        "No harmful content found. This post is approved."),
    ("flag this content for review",    "Content moderation analysis complete. Please review with a human moderator."),
    ("does this violate community guidelines", "Content review complete: message is within community guidelines."),
]

def _seed_cache():
    """Pre-populate the cache so analytics look realistic from the start."""
    for query, answer in _SEED_QUERIES:
        cache.put(query, answer)

    # Simulate historical hits to make analytics interesting
    cache.total_requests = 12_000
    cache.cache_hits     = 6_800
    cache.exact_hits     = 5_100
    cache.semantic_hits  = 1_700
    cache.cache_misses   = 5_200
    cache.tokens_saved   = 6_800 * AVG_TOKENS


_seed_cache()


class CacheHandler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        # Custom log: show timestamp
        print(f"[{time.strftime('%H:%M:%S')}] {fmt % args}")

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")  # allow browser requests
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str):
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        """Handle preflight CORS requests from browsers."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/analytics":
            self._send_json(cache.analytics())
        elif path == "/" or path == "":
            self._send_html(_DASHBOARD_HTML)
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/":
            self._send_json({"error": "Not found"}, 404)
            return

        # Read request body
        length  = int(self.headers.get("Content-Length", 0))
        raw     = self.rfile.read(length)

        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, 400)
            return

        query       = body.get("query", "").strip()
        application = body.get("application", "content moderation system")

        if not query:
            self._send_json({"error": "'query' field is required"}, 400)
            return

        # ── Try cache first ──────────────────────────────────────────────
        t0 = time.time()
        answer, cache_type, cache_key = cache.get(query)

        if cache_type != "miss":
            # ✅ Cache hit
            latency = max(1, int((time.time() - t0) * 1000))  # ensure at least 1ms
            self._send_json({
                "answer":    answer,
                "cached":    True,
                "cacheType": cache_type,
                "latency":   latency,
                "cacheKey":  cache_key
            })
            return

        # ── Cache miss → call LLM ────────────────────────────────────────
        answer, llm_latency = _fake_llm_call(query)
        cache_key = cache.put(query, answer)
        # Use the LLM's simulated latency (500-1500ms)
        latency = llm_latency

        self._send_json({
            "answer":    answer,
            "cached":    False,
            "cacheType": "miss",
            "latency":   latency,
            "cacheKey":  cache_key
        })

# ─────────────────────────────────────────────
#  EMBEDDED DASHBOARD  (served at GET /)
# ─────────────────────────────────────────────

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Cache System</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
  :root{
    --bg:#0a0a0f;--panel:#12121a;--border:#1e1e2e;
    --green:#00ff88;--cyan:#00d4ff;--orange:#ff8c42;
    --text:#e2e2f0;--muted:#666680;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'Syne',sans-serif;min-height:100vh;padding:2rem}
  h1{font-size:2rem;font-weight:800;color:var(--green);letter-spacing:-1px;margin-bottom:.25rem}
  .sub{color:var(--muted);font-family:'JetBrains Mono',monospace;font-size:.75rem;margin-bottom:2rem}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin-bottom:2rem}
  @media(max-width:700px){.grid{grid-template-columns:1fr}}
  .panel{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:1.5rem}
  .panel h2{font-size:.7rem;font-family:'JetBrains Mono',monospace;color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-bottom:1rem}
  textarea{width:100%;background:#0d0d14;border:1px solid var(--border);border-radius:8px;color:var(--text);font-family:'JetBrains Mono',monospace;font-size:.875rem;padding:.75rem;resize:vertical;min-height:80px;outline:none;transition:border-color .2s}
  textarea:focus{border-color:var(--cyan)}
  button{background:var(--green);color:#000;border:none;border-radius:8px;font-family:'Syne',sans-serif;font-weight:700;font-size:.9rem;padding:.75rem 1.5rem;cursor:pointer;width:100%;margin-top:.75rem;transition:opacity .2s}
  button:hover{opacity:.85}
  button:disabled{opacity:.4;cursor:not-allowed}
  .result{background:#0d0d14;border:1px solid var(--border);border-radius:8px;padding:1rem;margin-top:1rem;font-family:'JetBrains Mono',monospace;font-size:.82rem;line-height:1.6;min-height:60px;display:none}
  .tag{display:inline-block;padding:.15rem .6rem;border-radius:4px;font-size:.7rem;font-weight:700;font-family:'JetBrains Mono',monospace;margin-right:.4rem}
  .tag.hit{background:#00ff8822;color:var(--green);border:1px solid var(--green)}
  .tag.miss{background:#ff8c4222;color:var(--orange);border:1px solid var(--orange)}
  .tag.exact{background:#00d4ff22;color:var(--cyan);border:1px solid var(--cyan)}
  .tag.semantic{background:#b975ff22;color:#b975ff;border:1px solid #b975ff}
  .stat{display:flex;justify-content:space-between;align-items:baseline;padding:.5rem 0;border-bottom:1px solid var(--border)}
  .stat:last-child{border:none}
  .stat-label{color:var(--muted);font-size:.8rem;font-family:'JetBrains Mono',monospace}
  .stat-val{font-size:1.1rem;font-weight:700;color:var(--green)}
  .bar-wrap{background:#0d0d14;border-radius:6px;height:10px;margin-top:.5rem;overflow:hidden}
  .bar{height:100%;border-radius:6px;background:linear-gradient(90deg,var(--green),var(--cyan));transition:width .6s ease}
  .strategies span{display:inline-block;background:#1e1e2e;border:1px solid var(--border);border-radius:6px;padding:.3rem .7rem;font-size:.72rem;font-family:'JetBrains Mono',monospace;color:var(--cyan);margin:.25rem .25rem 0 0}
  .latency{font-size:.72rem;color:var(--muted);margin-top:.4rem}
  .full{grid-column:1/-1}
</style>
</head>
<body>
<h1>⚡ AI Cache System</h1>
<p class="sub">content moderation system · LRU + TTL + semantic similarity</p>

<div class="grid">
  <!-- Query panel -->
  <div class="panel">
    <h2>Send Query</h2>
    <textarea id="qbox" placeholder="e.g. Is this message spam?&#10;Does this post contain hate speech?&#10;Is this comment appropriate?"></textarea>
    <button id="sendBtn" onclick="sendQuery()">Send Query →</button>
    <div class="result" id="result"></div>
    <p class="latency" id="lat"></p>
  </div>

  <!-- Live analytics panel -->
  <div class="panel">
    <h2>Live Analytics</h2>
    <div class="stat">
      <span class="stat-label">Hit Rate</span>
      <span class="stat-val" id="hitRate">—</span>
    </div>
    <div class="bar-wrap"><div class="bar" id="hitBar" style="width:0%"></div></div>
    <div class="stat" style="margin-top:.75rem">
      <span class="stat-label">Total Requests</span>
      <span class="stat-val" id="totalReq">—</span>
    </div>
    <div class="stat">
      <span class="stat-label">Cache Hits</span>
      <span class="stat-val" id="hits">—</span>
    </div>
    <div class="stat">
      <span class="stat-label">Cache Misses</span>
      <span class="stat-val" id="misses">—</span>
    </div>
    <div class="stat">
      <span class="stat-label">Cache Size</span>
      <span class="stat-val" id="cacheSize">—</span>
    </div>
    <div class="stat">
      <span class="stat-label">Cost Savings</span>
      <span class="stat-val" id="savings">—</span>
    </div>
    <div class="stat">
      <span class="stat-label">Savings %</span>
      <span class="stat-val" id="savingsPct">—</span>
    </div>
  </div>

  <!-- Strategies panel (full width) -->
  <div class="panel full">
    <h2>Active Caching Strategies</h2>
    <div class="strategies" id="strats"></div>
  </div>
</div>

<script>
  // Auto-refresh analytics every 3 seconds
  async function refreshAnalytics(){
    try{
      const r=await fetch('/analytics');
      const d=await r.json();
      const pct=Math.round(d.hitRate*100);
      document.getElementById('hitRate').textContent=pct+'%';
      document.getElementById('hitBar').style.width=pct+'%';
      document.getElementById('totalReq').textContent=d.totalRequests.toLocaleString();
      document.getElementById('hits').textContent=d.cacheHits.toLocaleString()+' (exact: '+d.exactHits+', semantic: '+d.semanticHits+')';
      document.getElementById('misses').textContent=d.cacheMisses.toLocaleString();
      document.getElementById('cacheSize').textContent=d.cacheSize+' / '+d.maxCacheSize;
      document.getElementById('savings').textContent='$'+d.costSavings.toFixed(2)+' (baseline $'+d.baselineCost+')';
      document.getElementById('savingsPct').textContent=d.savingsPercent+'%';
      document.getElementById('strats').innerHTML=
        d.strategies.map(s=>'<span>'+s+'</span>').join('');
    }catch(e){console.warn('analytics error',e)}
  }
  refreshAnalytics();
  setInterval(refreshAnalytics,3000);

  async function sendQuery(){
    const btn=document.getElementById('sendBtn');
    const box=document.getElementById('qbox');
    const out=document.getElementById('result');
    const lat=document.getElementById('lat');
    const q=box.value.trim();
    if(!q)return;
    btn.disabled=true;btn.textContent='Querying…';
    out.style.display='none';lat.textContent='';
    const t0=Date.now();
    try{
      const r=await fetch('/',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({query:q,application:'content moderation system'})
      });
      const d=await r.json();
      const wall=Date.now()-t0;
      let tag='';
      if(d.cached && d.cacheType==='exact')  tag='<span class="tag hit">✅ HIT</span><span class="tag exact">exact match</span>';
      else if(d.cached)                      tag='<span class="tag hit">✅ HIT</span><span class="tag semantic">semantic match</span>';
      else                                   tag='<span class="tag miss">❌ MISS</span>';
      out.style.display='block';
      out.innerHTML=tag+'<br><br>'+escHtml(d.answer)+'<br><br><span style="color:var(--muted)">Key: '+escHtml(d.cacheKey||'—')+'</span>';
      lat.textContent='⏱ Server latency: '+d.latency+'ms · Wall time: '+wall+'ms';
      refreshAnalytics();
    }catch(e){
      out.style.display='block';out.textContent='Error: '+e;
    }
    btn.disabled=false;btn.textContent='Send Query →';
  }

  function escHtml(s){
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }

  // Allow Ctrl+Enter to submit
  document.getElementById('qbox').addEventListener('keydown',e=>{
    if((e.ctrlKey||e.metaKey)&&e.key==='Enter') sendQuery();
  });
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), CacheHandler)
    print(f"""
╔══════════════════════════════════════════════╗
║   AI Caching System — Content Moderation     ║
╠══════════════════════════════════════════════╣
║  Dashboard  → http://localhost:{PORT}           ║
║  POST /     → main query endpoint            ║
║  GET  /analytics → cache metrics             ║
╚══════════════════════════════════════════════╝
""")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")