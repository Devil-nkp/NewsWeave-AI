import os
import json
import uvicorn
import re
import time
import logging
import asyncio
import httpx 
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date
from fastapi import FastAPI, Request, Depends, Response
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS 
import wikipedia
import pandas as pd
import plotly.express as px
import plotly.utils
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NewsWeave-Supreme")

INTERNAL_API_KEY = os.environ.get("GROQ_API_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL")

if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# --- DATABASE SETUP ---
Base = declarative_base()
class GlobalStats(Base):
    __tablename__ = "global_stats"
    id = Column(Integer, primary_key=True, index=True)
    total_likes = Column(Integer, default=0)
    total_prompts = Column(Integer, default=0)
    prompts_today = Column(Integer, default=0)
    last_date = Column(String, default=str(date.today()))

if DATABASE_URL:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
else:
    engine = None
    SessionLocal = None

def get_db():
    if SessionLocal:
        db = SessionLocal()
        try: yield db
        finally: db.close()
    else: yield None

app = FastAPI()
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

TRENDING_CACHE = {} 

def get_or_create_stats(db: Session):
    if not db: return None
    stats = db.query(GlobalStats).first()
    if not stats:
        stats = GlobalStats(id=1, total_likes=0, total_prompts=0, prompts_today=0, last_date=str(date.today()))
        db.add(stats)
        db.commit()
    if stats.last_date != str(date.today()):
        stats.prompts_today = 0
        stats.last_date = str(date.today())
        db.commit()
    return stats

DISCLAIMER_HTML = """
<div style="background: rgba(0, 243, 255, 0.05); border-left: 3px solid #00f3ff; padding: 15px; margin-bottom: 25px; border-radius: 4px; font-size: 0.9rem; color: #aeeeff; display: flex; align-items: center; gap: 12px; box-shadow: 0 0 15px rgba(0, 243, 255, 0.1);">
    <i class="fas fa-shield-alt" style="font-size:1.2rem;"></i>
    <div><strong>FORENSIC AI REPORT:</strong> Data generated autonomously. Verify critical intelligence with primary sources.</div>
</div>
"""

REGION_MAP = ["Global"] + sorted([
    "Argentina", "Australia", "Austria", "Belgium (FR)", "Belgium (NL)", "Brazil", "Bulgaria",
    "Canada (EN)", "Canada (FR)", "Chile", "China", "Colombia", "Croatia", "Czech Republic", "Denmark",
    "Egypt", "Estonia", "Finland", "France", "Germany", "Greece", "Hong Kong", "Hungary", "India",
    "Indonesia", "Ireland", "Israel", "Italy", "Japan", "Korea", "Latvia", "Lithuania", "Malaysia",
    "Mexico", "Netherlands", "New Zealand", "Norway", "Pakistan", "Peru", "Philippines", "Poland",
    "Portugal", "Romania", "Russia", "Saudi Arabia", "Singapore", "Slovakia", "Slovenia", "South Africa",
    "Spain", "Sweden", "Switzerland (DE)", "Switzerland (FR)", "Taiwan", "Thailand", "Turkey", "UK",
    "Ukraine", "USA", "Vietnam"
])

CATEGORY_IMAGES = {
    "tech": "https://images.unsplash.com/photo-1518770660439-4636190af475?w=600&q=80",
    "finance": "https://images.unsplash.com/photo-1611974765270-ca1258634369?w=600&q=80",
    "war": "https://images.unsplash.com/photo-1557426272-fc759fdf7a8d?w=600&q=80",
    "politics": "https://images.unsplash.com/photo-1555848962-6e79363ec58f?w=600&q=80",
    "health": "https://images.unsplash.com/photo-1532938911079-1b06ac7ceec7?w=600&q=80",
    "general": "https://images.unsplash.com/photo-1504711434969-e33886168f5c?w=600&q=80"
}

class SearchRequest(BaseModel):
    topic: str
    region: str
    mode: str

class TrendingRequest(BaseModel):
    region: str

class SwarmCommander:
    def __init__(self):
        self.llm = ChatGroq(temperature=0.3, model_name="llama-3.3-70b-versatile", api_key=INTERNAL_API_KEY) if INTERNAL_API_KEY else None

    def _hunter_agent(self, topic, region):
        reg = "wt-wt"
        if region == "India": reg = "in-en"
        elif region == "USA": reg = "us-en"
        
        vault = ""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(f"{topic} news analysis {datetime.now().year}", region=reg, max_results=12))
                for r in results:
                    vault += f"SOURCE: {r['title']}\nLINK: {r['href']}\nCONTENT: {r['body']}\n\n"
        except: pass
        
        if len(vault) < 500:
            try: vault += f"WIKI-SUMMARY: {wikipedia.summary(topic, sentences=8)}"
            except: pass
            
        return vault if vault else "No specific data found via Live Search."

    def _vision_agent(self, topic, region):
        gallery = []
        try:
            with DDGS() as ddgs:
                results = list(ddgs.images(f"{topic} news photo", region="wt-wt", max_results=40))
                for r in results:
                    if len(gallery) >= 25: break
                    if r['image'] and r['image'].startswith('http'):
                        gallery.append({"src": r['image'], "title": r['title']})
        except: pass
        return gallery

    def _analyst_agent(self, text):
        try:
            years = re.findall(r'\b(20\d{2})\b', text)
            nums = re.findall(r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\b', text)
            clean_years, clean_nums = [], []
            if len(years) > 1 and len(nums) > 1:
                for i in range(min(len(years), len(nums))):
                    try:
                        y = int(years[i])
                        v = float(nums[i].replace(',', ''))
                        if 1950 < v < 2100: continue
                        clean_years.append(y); clean_nums.append(v)
                    except: pass
                
                if clean_years:
                    df = pd.DataFrame({"Year": clean_years, "Metric": clean_nums}).sort_values('Year')
                    df = df.groupby('Year', as_index=False).mean()
                    fig = px.area(df, x="Year", y="Metric", title=f"Trend Analysis: {datetime.now().year}", template="plotly_dark", markers=True)
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#aeeeff"), autosize=True)
                    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        except: pass
        return None

    def execute(self, topic, region, mode):
        context = self._hunter_agent(topic, region)
        images = self._vision_agent(topic, region)
        
        prompt = f"""
        You are NewsWeave Supreme v46. Produce a HIGH-LEVEL INTELLIGENCE REPORT.
        TOPIC: {topic} | REGION: {region} | MODE: {mode}
        CONTEXT: {context}
        INSTRUCTIONS:
        1. Write a LONG, DETAILED report (minimum 800 words).
        2. Use the following structure strictly:
           - <h2>EXECUTIVE SUMMARY</h2>: A high-level strategic overview.
           - <h2>KEY FINDINGS & METRICS</h2>: Bullet points with specific numbers/dates.
           - <h2>DEEP DIVE ANALYSIS</h2>: Multi-paragraph detailed breakdown.
           - <h2>GEOPOLITICAL/MARKET IMPACT</h2>: How this affects the chosen region vs global.
           - <h2>SOURCES</h2>: List cited domains.
        3. Tone: Professional, Objective, Forensic.
        4. Use HTML tags (h2, p, ul, li, strong) for formatting.
        """
        try: report = self.llm.invoke(prompt).content if self.llm else "LLM Offline."
        except Exception as e: report = f"Analysis Interrupted: {str(e)}"

        chart = self._analyst_agent(report + context)
        final_html = DISCLAIMER_HTML + report
        return final_html, images, chart

agent = SwarmCommander()
semaphore = asyncio.Semaphore(3)
executor = ThreadPoolExecutor(max_workers=6)

@app.get("/")
async def index(request: Request, db: Session = Depends(get_db)):
    count = 0
    if db:
        s = get_or_create_stats(db)
        count = s.total_likes
    return templates.TemplateResponse("index.html", {"request": request, "regions": REGION_MAP, "total_likes": count})

@app.get("/proxy-image")
async def proxy_image(url: str):
    """Fetches external images to bypass CORS in PDF generation."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            content_type = resp.headers.get("content-type", "image/jpeg")
            return Response(content=resp.content, media_type=content_type)
    except:
        return Response(status_code=404)

@app.post("/analyze")
async def analyze(request: SearchRequest, db: Session = Depends(get_db)):
    if db:
        s = get_or_create_stats(db)
        s.total_prompts += 1
        db.commit()
    report, images, chart = agent.execute(request.topic, request.region, request.mode)
    return JSONResponse({"report": report, "images": images, "chart": chart})

@app.post("/like")
async def like(db: Session = Depends(get_db)):
    count = 0
    if db:
        s = get_or_create_stats(db)
        s.total_likes += 1
        db.commit()
        db.refresh(s)
        count = s.total_likes
    return JSONResponse({"new_count": count})

async def fetch_img(title):
    async with semaphore:
        loop = asyncio.get_event_loop()
        def s():
            try:
                with DDGS() as ddgs:
                    r = list(ddgs.images(f"{title} news", max_results=1))
                    return r[0]['image'] if r else None
            except: return None
        return await loop.run_in_executor(executor, s)

@app.post("/trending")
async def trending(request: TrendingRequest):
    if request.region in TRENDING_CACHE: return JSONResponse({"headlines": TRENDING_CACHE[request.region]})
    try:
        loop = asyncio.get_event_loop()
        def get_n():
            reg = "wt-wt"
            with DDGS() as d: return list(d.news(f"top news {request.region}", region=reg, max_results=8))
        raw = await loop.run_in_executor(executor, get_n)
        tasks = []
        headlines = []
        for r in raw:
            h = {"title": r['title'], "source": r['source'], "date": r['date'], "image": r.get('image')}
            headlines.append(h)
            if not h['image']: tasks.append(fetch_img(r['title']))
            else: tasks.append(asyncio.sleep(0, result=h['image']))
        imgs = await asyncio.gather(*tasks)
        for i, url in enumerate(imgs):
            if not headlines[i]['image']:
                t = headlines[i]['title'].lower()
                if "tech" in t: fallback = CATEGORY_IMAGES['tech']
                elif "market" in t: fallback = CATEGORY_IMAGES['finance']
                else: fallback = CATEGORY_IMAGES['general']
                headlines[i]['image'] = url or fallback
        TRENDING_CACHE[request.region] = headlines
        return JSONResponse({"headlines": headlines})
    except:
        return JSONResponse({"headlines": []})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
