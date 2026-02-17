# 🚀 AI Caching System - Content Moderation

A high-performance caching system for AI-powered content moderation that reduces API costs by 48%+ using intelligent caching strategies.

## ⚡ Features

- **Exact Match Caching** — MD5 hash-based cache for identical queries
- **Semantic Caching** — Cosine similarity matching (≥0.95) for similar queries  
- **LRU Eviction** — Automatically removes least-recently-used entries
- **TTL Expiration** — 24-hour time-to-live for fresh results
- **Real-time Analytics** — Live dashboard with cost savings tracking

## 🎯 Performance Metrics

- **Cache Hit Latency:** <50ms
- **Cache Miss Latency:** <2000ms  
- **Target Hit Rate:** 48%+
- **Daily Cost Savings:** $2.00+

## 🏃 Quick Start

### Run Locally

```bash
python3 cache_server.py
```

Then open http://localhost:8080

### API Endpoints

**POST /** - Query the AI system
```bash
curl -X POST http://localhost:8080/ \
  -H "Content-Type: application/json" \
  -d '{"query": "Is this spam?", "application": "content moderation system"}'
```

Response:
```json
{
  "answer": "This content has been flagged as potential spam...",
  "cached": true,
  "cacheType": "exact",
  "latency": 2,
  "cacheKey": "a3f82bc1..."
}
```

**GET /analytics** - View cache performance
```bash
curl http://localhost:8080/analytics
```

Response:
```json
{
  "hitRate": 0.5667,
  "totalRequests": 12003,
  "cacheHits": 6802,
  "cacheMisses": 5201,
  "cacheSize": 15,
  "costSavings": 2.45,
  "savingsPercent": 46.4,
  "strategies": [
    "exact match caching",
    "semantic similarity (cosine ≥ 0.95)",
    "LRU eviction policy",
    "TTL expiration (24h)"
  ]
}
```

## 🌐 Deploy

### Render

1. Fork this repo
2. Create a new Web Service on Render
3. Connect your GitHub repo
4. Render auto-deploys!

### Railway

1. Fork this repo  
2. Create new project on Railway
3. Connect GitHub → Deploy automatically

### Replit

1. Import from GitHub
2. Click Run → Instant deployment

## 📊 Testing

See [TESTING_GUIDE.md](TESTING_GUIDE.md) for detailed test scenarios.

Quick test:
1. Send a query → See MISS (slow)
2. Repeat query → See EXACT HIT (fast)  
3. Similar query → See SEMANTIC HIT (fast)
4. Check /analytics → See improving hit rate

## 🛠️ Configuration

Edit `cache_server.py`:

```python
MAX_CACHE_SIZE = 1500          # Max entries before eviction
TTL_SECONDS = 86_400           # 24 hours
SIMILARITY_THRESHOLD = 0.95    # Semantic match cutoff
PORT = 8080                    # Server port
```

## 📝 Assignment Requirements

✅ Exact match caching  
✅ Semantic caching (embedding similarity > 0.95)  
✅ LRU eviction policy  
✅ TTL expiration (24 hours)  
✅ Cache hits <50ms  
✅ Cache misses <2000ms  
✅ POST / endpoint with JSON response  
✅ GET /analytics endpoint  
✅ Cost savings tracking  

## 🧠 How It Works

1. **Query arrives** → Normalize text ("Hello" → "hello")
2. **Check exact match** → MD5 hash lookup (instant)
3. **Check semantic match** → Cosine similarity scan (fast)
4. **Cache miss?** → Call LLM, cache result
5. **LRU eviction** → Remove oldest entries when cache is full
6. **TTL check** → Expire entries older than 24 hours

## 📖 Documentation

- [QUICK_START.md](QUICK_START.md) — Get started in 3 steps
- [TESTING_GUIDE.md](TESTING_GUIDE.md) — Comprehensive testing guide

## 🎓 Assignment Context

**Application:** Content Moderation System  
**Daily Requests:** 43,972  
**Cacheable Queries:** ~65% (28,581 repeating)  
**Model Cost:** $0.40 per 1M tokens  
**Baseline Cost:** $5.28/day  
**Target Savings:** $2.00+/day (48%+ hit rate)

## 📄 License

Educational project for AI system optimization assignment.

---

**Live Demo:** [Your deployed URL here]
