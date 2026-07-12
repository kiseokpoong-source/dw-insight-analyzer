import asyncio
import json
import os
import re
from typing import List

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="DW Insight Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
async_client = anthropic.AsyncAnthropic(api_key=_api_key)


class ParseRequest(BaseModel):
    text: str


class AnalyzeRequest(BaseModel):
    titles: List[str]


# ── Title extraction ──────────────────────────────────────────────────────────

def _is_title_line(line: str) -> bool:
    if not line or len(line) < 6:
        return False
    if re.match(r"^\d+$", line):
        return False
    if re.match(r"^\d+\.?\d*[KkMmBb]$", line):
        return False
    if "·" in line:
        return False
    if re.match(r"^[A-Za-z][^:]{1,40}:\s+\S", line):
        return False
    if len(line) > 40 and line[0].islower():
        return False
    return line[0].isupper()


def extract_titles(raw: str) -> List[str]:
    seen: set = set()
    results: List[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if line and _is_title_line(line) and line not in seen:
            seen.add(line)
            results.append(line)
    return results


# ── Anthropic analysis ────────────────────────────────────────────────────────

_SYSTEM = """You are a car detailing market intelligence analyst for Korean business reporting.

CRITICAL RULE: ALL descriptive text in your JSON MUST be written in Korean (한국어).
This includes: insights, keywords, complaints, requirements, gap_position, categories.
Only brand/product names keep their original English spelling (e.g. Meguiar's, AutoGlym, Koch Chemie).

Respond with ONLY a valid JSON object — no markdown fences, no preamble, no trailing text."""

_PROMPT_TEMPLATE = """Search Detailing World for this forum thread: "{title}"

Return ONLY this exact JSON structure (all text values in Korean):
{{
  "brands": [],
  "categories": [],
  "insights": [],
  "keywords": [],
  "complaints_top": [],
  "requirements_top": [],
  "sentiment": {{"positive": 0, "negative": 0, "neutral": 0}},
  "comparisons": [],
  "value_mentions": false,
  "price_sensitive": false,
  "korean_brands": [],
  "year_mentioned": null,
  "gap_position": ""
}}

Field rules:
- brands: list of car care brand names (original spelling)
- categories: Korean detailing categories (세차/광택/왁스/실런트/세라믹코팅/도장보호/실내/휠/타이어/기타)
- insights: 2-3 key discussion points IN KOREAN
- keywords: 5-8 technical keywords IN KOREAN
- complaints_top: up to 5 items as [{{"complaint": "한국어 불만내용", "count": N}}]
- requirements_top: up to 5 items as [{{"requirement": "한국어 요구사항", "count": N}}]
- sentiment: estimated positive/negative/neutral mention counts as integers 0-10
- comparisons: products users directly compare [{{"product_a": "브랜드A", "product_b": "브랜드B"}}]
- value_mentions: true if value-for-money or cost-effectiveness is discussed
- price_sensitive: true if price is a primary concern in the thread
- korean_brands: Korean-origin car care brands only (e.g. 불곰, 크리스탈, K2)
- year_mentioned: integer year referenced in thread or null if none
- gap_position: Korean description of an unmet market need found in discussion, empty string if none"""


def _build_prompt(title: str) -> str:
    return _PROMPT_TEMPLATE.format(title=title)


def _parse_json(text: str) -> dict:
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}


async def _analyze_one(title: str) -> dict:
    try:
        resp = await async_client.messages.create(
            model="claude-opus-4-8",
            max_tokens=2048,
            system=_SYSTEM,
            tools=[{
                "type": "web_search_20260209",
                "name": "web_search",
                "max_uses": 2,
                "allowed_domains": ["detailingworld.co.uk"],
            }],
            messages=[{"role": "user", "content": _build_prompt(title)}],
        )
        text = "".join(
            b.text for b in resp.content if hasattr(b, "text") and b.text
        )
        data = _parse_json(text)
        return {
            "title": title,
            "brands": data.get("brands", []),
            "categories": data.get("categories", []),
            "insights": data.get("insights", []),
            "keywords": data.get("keywords", []),
            "complaints_top": data.get("complaints_top", []),
            "requirements_top": data.get("requirements_top", []),
            "sentiment": data.get("sentiment", {"positive": 0, "negative": 0, "neutral": 0}),
            "comparisons": data.get("comparisons", []),
            "value_mentions": data.get("value_mentions", False),
            "price_sensitive": data.get("price_sensitive", False),
            "korean_brands": data.get("korean_brands", []),
            "year_mentioned": data.get("year_mentioned"),
            "gap_position": data.get("gap_position", ""),
        }
    except Exception as exc:
        return {
            "title": title,
            "brands": [], "categories": [], "insights": [f"분석 오류: {str(exc)[:120]}"],
            "keywords": [], "complaints_top": [], "requirements_top": [],
            "sentiment": {"positive": 0, "negative": 0, "neutral": 0},
            "comparisons": [], "value_mentions": False, "price_sensitive": False,
            "korean_brands": [], "year_mentioned": None, "gap_position": "",
            "error": True,
        }


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.post("/api/parse")
async def api_parse(req: ParseRequest):
    titles = extract_titles(req.text)
    return {"titles": titles, "count": len(titles)}


@app.post("/api/analyze")
async def api_analyze(req: AnalyzeRequest):
    if not req.titles:
        raise HTTPException(status_code=400, detail="titles 목록이 비어있습니다.")
    if not _api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY가 설정되지 않았습니다.")

    titles = req.titles[:20]
    sem = asyncio.Semaphore(3)

    async def bounded(t: str) -> dict:
        async with sem:
            return await _analyze_one(t)

    results = await asyncio.gather(*[bounded(t) for t in titles])
    return {"results": list(results)}


# ── Static files ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
