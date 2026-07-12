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
    # number-only
    if re.match(r"^\d+$", line):
        return False
    # view-count like 2.7K, 1.2M
    if re.match(r"^\d+\.?\d*[KkMmBb]$", line):
        return False
    # author · date
    if "·" in line:
        return False
    # tag lines: "Pre-wash stages: ...", "Wash: ..."
    if re.match(r"^[A-Za-z][^:]{1,40}:\s+\S", line):
        return False
    # long preview text starting with lowercase
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

_SYSTEM = (
    "You are a car detailing data analyst. "
    "Respond with ONLY a valid JSON object — no markdown, no preamble, no extra text. "
    "Extract: brands (car care product brand names), "
    "categories (detailing categories: wash / polish / wax / sealant / "
    "ceramic coating / paint correction / interior / wheels / tyres / vinyl wrap / etc.), "
    "insights (2-3 key discussion points from detailers), "
    "keywords (5-8 relevant technical terms)."
)

_PROMPT = (
    'Search Detailing World for this thread: "{title}"\n\n'
    "Return ONLY this JSON object (no other text):\n"
    '{{"brands":[],"categories":[],"insights":[],"keywords":[]}}'
)


def _parse_json(text: str) -> dict:
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find a JSON object in the response
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
            max_tokens=1024,
            system=_SYSTEM,
            tools=[{
                "type": "web_search_20260209",
                "name": "web_search",
                "max_uses": 2,
                "allowed_domains": ["detailingworld.co.uk"],
            }],
            messages=[{"role": "user", "content": _PROMPT.format(title=title)}],
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
        }
    except Exception as exc:
        return {
            "title": title,
            "brands": [],
            "categories": [],
            "insights": [f"분석 오류: {str(exc)[:120]}"],
            "keywords": [],
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
