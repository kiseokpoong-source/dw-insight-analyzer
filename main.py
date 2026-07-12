import json
import os
import re
from typing import List

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
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


# ── /api/parse — Claude API 기반 제목 추출 ───────────────────────────────────

_PARSE_PROMPT = """다음은 Detailing World 포럼 검색결과를 복사한 텍스트야.
여기서 포럼 스레드 제목만 정확히 추출해서 JSON 배열로 반환해줘.
제목 아닌 것: Pre-wash stages 카테고리, 미리보기 텍스트, 조회수, 댓글수, 작성자, 날짜, 브랜드명 단독
반드시 JSON 배열만 반환: ["제목1", "제목2", ...]

텍스트:
{text}"""


def _parse_titles_json(text: str) -> List[str]:
    if not text:
        return []
    cleaned = text.strip()
    # Remove markdown code fences if present
    cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```$", "", cleaned)
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [str(t).strip() for t in data if t and str(t).strip()]
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[.*?\]", cleaned, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            if isinstance(data, list):
                return [str(t).strip() for t in data if t and str(t).strip()]
        except json.JSONDecodeError:
            pass
    return []


@app.post("/api/parse")
async def api_parse(req: ParseRequest):
    if not req.text.strip():
        return {"titles": [], "count": 0}
    if not _api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY가 설정되지 않았습니다.")

    resp = await async_client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": _PARSE_PROMPT.format(text=req.text),
        }],
    )

    raw = resp.content[0].text if resp.content else ""
    titles = _parse_titles_json(raw)
    return {"titles": titles, "count": len(titles)}


# ── /api/analyze — SSE 스트리밍 (순차 처리) ──────────────────────────────────

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
- brands: car care brand names (original spelling)
- categories: Korean (세차/광택/왁스/실런트/세라믹코팅/도장보호/실내/휠/타이어/기타)
- insights: 2-3 key points IN KOREAN
- keywords: 5-8 terms IN KOREAN
- complaints_top: up to 5 [{{"complaint": "한국어", "count": N}}]
- requirements_top: up to 5 [{{"requirement": "한국어", "count": N}}]
- sentiment: integer scores 0-10
- comparisons: [{{"product_a": "A", "product_b": "B"}}]
- value_mentions: true if value-for-money discussed
- price_sensitive: true if price is a key concern
- korean_brands: Korean-origin brands only
- year_mentioned: integer year or null
- gap_position: Korean description of unmet need, empty string if none"""


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
            messages=[{
                "role": "user",
                "content": _PROMPT_TEMPLATE.format(title=title),
            }],
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


@app.post("/api/analyze")
async def api_analyze(req: AnalyzeRequest):
    if not req.titles:
        raise HTTPException(status_code=400, detail="titles 목록이 비어있습니다.")
    if not _api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY가 설정되지 않았습니다.")

    titles = req.titles[:20]

    async def event_stream():
        for title in titles:
            result = await _analyze_one(title)
            payload = json.dumps(result, ensure_ascii=False)
            yield f"data: {payload}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # disable nginx/Railway proxy buffering
        },
    )


# ── Static files ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        timeout_keep_alive=120,
    )
