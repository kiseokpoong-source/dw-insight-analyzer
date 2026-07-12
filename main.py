import asyncio
import json
import os
import re
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── DB ────────────────────────────────────────────────────────────────────────

DB_PATH = "results.db"

def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS analyzed_results (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        analyzed_at   TEXT NOT NULL,
        title         TEXT NOT NULL,
        brands        TEXT,
        categories    TEXT,
        insights      TEXT,
        keywords      TEXT,
        perf_kw       TEXT,
        req_kw        TEXT,
        comp_kw       TEXT,
        complaints    TEXT,
        requirements  TEXT,
        comparisons   TEXT,
        value_mentions INTEGER DEFAULT 0,
        price_sensitive INTEGER DEFAULT 0,
        korean_brands TEXT,
        year_mentioned INTEGER,
        gap_position  TEXT
    )""")
    conn.commit()
    conn.close()

def _save_to_db(results: list):
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(DB_PATH)
    j = lambda v: json.dumps(v, ensure_ascii=False)
    for r in results:
        conn.execute("""
        INSERT INTO analyzed_results
        (analyzed_at,title,brands,categories,insights,keywords,perf_kw,req_kw,comp_kw,
         complaints,requirements,comparisons,value_mentions,price_sensitive,
         korean_brands,year_mentioned,gap_position)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
            now, r.get("title",""),
            j(r.get("brands",[])), j(r.get("categories",[])),
            j(r.get("insights",[])), j(r.get("keywords",[])),
            j(r.get("performance_keywords",[])), j(r.get("requirement_keywords",[])),
            j(r.get("complaint_keywords",[])), j(r.get("complaints_top",[])),
            j(r.get("requirements_top",[])), j(r.get("comparisons",[])),
            1 if r.get("value_mentions") else 0,
            1 if r.get("price_sensitive") else 0,
            j(r.get("korean_brands",[])),
            r.get("year_mentioned"),
            r.get("gap_position",""),
        ))
    conn.commit()
    conn.close()


# ── App ───────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_db()
    yield

app = FastAPI(title="DW Insight Analyzer", lifespan=lifespan)

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
    cleaned = re.sub(r"^```[a-z]*\n?", "", text.strip())
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
        timeout=120.0,
        messages=[{"role": "user", "content": _PARSE_PROMPT.format(text=req.text)}],
    )
    raw = resp.content[0].text if resp.content else ""
    titles = _parse_titles_json(raw)
    return {"titles": titles, "count": len(titles)}


# ── /api/analyze — 배치 처리 + SSE 스트리밍 ──────────────────────────────────

_BATCH_SYSTEM = """You are a car detailing market intelligence analyst for Korean business reporting.

CRITICAL: ALL descriptive text MUST be written in Korean (한국어).
Only brand/product names keep their original English spelling (e.g. Meguiar's, AutoGlym).
Respond with ONLY a valid JSON array — no markdown fences, no preamble."""

def _build_batch_prompt(titles: List[str]) -> str:
    numbered = "\n".join(f'{i+1}. "{t}"' for i, t in enumerate(titles))
    return f"""다음 {len(titles)}개의 Detailing World 포럼 스레드를 각각 site:detailingworld.co.uk 에서 검색하고 분석해줘.

분석할 스레드:
{numbered}

반드시 {len(titles)}개 객체를 포함한 JSON 배열만 반환 (한국어 텍스트):
[
  {{
    "title": "원본 스레드 제목 (변경 없이 그대로)",
    "brands": [],
    "categories": [],
    "insights": [],
    "keywords": [],
    "performance_keywords": [],
    "requirement_keywords": [],
    "complaint_keywords": [],
    "complaints_top": [],
    "requirements_top": [],
    "comparisons": [],
    "value_mentions": false,
    "price_sensitive": false,
    "korean_brands": [],
    "year_mentioned": null,
    "gap_position": ""
  }}
]

필드 규칙 (모든 텍스트는 한국어):
- brands: 제품 브랜드명 (영문 원문 유지)
- categories: 세차/광택/왁스/실런트/세라믹코팅/도장보호/실내/휠/타이어/기타
- insights: 핵심 토론 포인트 2-3개
- keywords: 일반 키워드 5-8개
- performance_keywords: 성능·특성 관련 키워드 (예: 내구성, 발수성, 광택도, 도포성)
- requirement_keywords: 사용자 요구사항 관련 키워드 (예: 쉬운 적용, 비용 절감, 안전한 성분)
- complaint_keywords: 불만 관련 키워드 (예: 제거 어려움, 짧은 지속성, 얼룩, 비싼 가격)
- complaints_top: 불만사항 top 5 [{{"complaint":"내용","count":N}}]
- requirements_top: 요구사항 top 5 [{{"requirement":"내용","count":N}}]
- comparisons: 비교된 제품 쌍 [{{"product_a":"A","product_b":"B"}}]
- value_mentions: 가성비 언급 시 true
- price_sensitive: 가격 민감도 높을 시 true
- korean_brands: 한국 브랜드만 (예: 불곰, 크리스탈)
- year_mentioned: 언급 연도 정수 또는 null
- gap_position: 미충족 시장 수요 설명, 없으면 빈 문자열"""


def _parse_json_array(text: str) -> List[dict]:
    if not text:
        return []
    cleaned = re.sub(r"^```[a-z]*\n?", "", text.strip())
    cleaned = re.sub(r"\n?```$", "", cleaned)
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
        except json.JSONDecodeError:
            pass
    return []


def _normalize_result(title: str, item: dict) -> dict:
    return {
        "title": title,
        "brands":               item.get("brands", []),
        "categories":           item.get("categories", []),
        "insights":             item.get("insights", []),
        "keywords":             item.get("keywords", []),
        "performance_keywords": item.get("performance_keywords", []),
        "requirement_keywords": item.get("requirement_keywords", []),
        "complaint_keywords":   item.get("complaint_keywords", []),
        "complaints_top":       item.get("complaints_top", []),
        "requirements_top":     item.get("requirements_top", []),
        "comparisons":          item.get("comparisons", []),
        "value_mentions":       bool(item.get("value_mentions", False)),
        "price_sensitive":      bool(item.get("price_sensitive", False)),
        "korean_brands":        item.get("korean_brands", []),
        "year_mentioned":       item.get("year_mentioned"),
        "gap_position":         item.get("gap_position", ""),
    }


async def _analyze_batch(titles: List[str]) -> List[dict]:
    try:
        resp = await async_client.messages.create(
            model="claude-opus-4-8",
            max_tokens=4096,
            timeout=120.0,
            system=_BATCH_SYSTEM,
            tools=[{
                "type": "web_search_20260209",
                "name": "web_search",
                "max_uses": len(titles) * 2,
                "allowed_domains": ["detailingworld.co.uk"],
            }],
            messages=[{"role": "user", "content": _build_batch_prompt(titles)}],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text") and b.text)
        items = _parse_json_array(text)

        # Match by title (exact → case-insensitive → index fallback)
        by_title = {it.get("title", ""): it for it in items}
        results = []
        for i, title in enumerate(titles):
            item = (
                by_title.get(title)
                or by_title.get(title.strip())
                or next((v for k, v in by_title.items() if k.lower() == title.lower()), None)
                or (items[i] if i < len(items) else {})
            )
            results.append(_normalize_result(title, item))
        return results

    except Exception as exc:
        return [_normalize_result(t, {"insights": [f"배치 분석 오류: {str(exc)[:80]}"]}) for t in titles]


@app.post("/api/analyze")
async def api_analyze(req: AnalyzeRequest):
    if not req.titles:
        raise HTTPException(status_code=400, detail="titles 목록이 비어있습니다.")
    if not _api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY가 설정되지 않았습니다.")

    titles = req.titles[:20]
    batch_size = 5
    batches = [titles[i:i+batch_size] for i in range(0, len(titles), batch_size)]

    async def event_stream():
        all_results = []
        for batch in batches:
            # Start batch task; send keep-alive every 5s while waiting
            task = asyncio.create_task(_analyze_batch(batch))
            while not task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"

            batch_results = task.result()
            all_results.extend(batch_results)
            for result in batch_results:
                yield f"data: {json.dumps(result, ensure_ascii=False)}\n\n"

        # Save all to SQLite (non-blocking)
        try:
            await asyncio.to_thread(_save_to_db, all_results)
        except Exception:
            pass
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
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
