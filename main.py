import os
import json
import uvicorn
import re
import time
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date
from fastapi import FastAPI, Request, Depends
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

# ENVIRONMENT VARIABLES
INTERNAL_API_KEY = os.environ.get("GROQ_API_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL") # Render provides this automatically

# Fix for Render's Postgres URL (starts with postgres://, needs postgresql://)
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# --- POSTGRESQL DATABASE SETUP ---
Base = declarative_base()

class GlobalStats(Base):
    __tablename__ = "global_stats"
    id = Column(Integer, primary_key=True, index=True)
    total_likes = Column(Integer, default=0)
    total_prompts = Column(Integer, default=0)
    prompts_today = Column(Integer, default=0)
    last_date = Column(String, default=str(date.today()))

# Create Engine
if DATABASE_URL:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    # Create Tables
    Base.metadata.create_all(bind=engine)
else:
    engine = None
    SessionLocal = None
    logger.warning("DATABASE_URL not found. Persistence disabled.")

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- APP INIT ---
app = FastAPI()
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# IN-MEMORY CACHE FOR TRENDING (Speed)
TRENDING_CACHE = {} 

# --- DB HELPERS ---
def get_or_create_stats(db: Session):
    stats = db.query(GlobalStats).first()
    if not stats:
        stats = GlobalStats(id=1, total_likes=0, total_prompts=0, prompts_today=0, last_date=str(date.today()))
        db.add(stats)
        db.commit()
        db.refresh(stats)
    
    # Check date reset
    if stats.last_date != str(date.today()):
        stats.prompts_today = 0
        stats.last_date = str(date.today())
        db.commit()
        db.refresh(stats)
        
    return stats

# --- CONSTANTS & DATA ---
DISCLAIMER_HTML = """
<div style="background: rgba(0, 243, 255, 0.05); border-left: 3px solid #00f3ff; padding: 15px; margin-bottom: 25px; border-radius: 4px; font-size: 0.9rem; color: #aeeeff; display: flex; align-items: center; gap: 12px; box-shadow: 0 0 15px rgba(0, 243, 255, 0.1);">
    <i class="fas fa-shield-alt" style="font-size:1.2rem;"></i>
    <div><strong>FORENSIC AI REPORT:</strong> Data generated autonomously. Verify critical intelligence with primary sources.</div>
</div>
"""

REGION_MAP = sorted([
    "Global", "Argentina", "Australia", "Austria", "Belgium (FR)", "Belgium (NL)", "Brazil", "Bulgaria",
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

# --- AGENT LOGIC ---
class SwarmCommander:
    def __init__(self):
        self.llm = ChatGroq(temperature=0.0, model_name="llama-3.3-70b-versatile", api_key=INTERNAL_API_KEY) if INTERNAL_API_KEY else None

    def _hunter(self, topic, region):
        reg_code = "wt-wt" # Default
        if region == "India": reg_code = "in-en"
        elif region == "USA": reg_code = "us-en"
        # Add simpler mapping logic or rely on semantic search
        
        vault = ""
        with DDGS() as ddgs:
            try:
                # Burst search
                results = list(ddgs.text(f"{topic} news", region=reg_code, max_results=4))
                for r in results: vault += f"SOURCE: {r['title']}\nLINK: {r['href']}\nDATA: {r['body']}\n\n"
            except: pass
        
        if not vault:
            try: vault = f"ENCYCLOPEDIA: {wikipedia.summary(topic, sentences=4)}"
            except: pass
            
        return vault if vault else "No data found."

    def _analyst(self, text):
        try:
            years = re.findall(r'\b(20\d{2})\b', text)
            nums = re.findall(r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\b', text)
            if len(years) > 1 and len(nums) > 1:
                df = pd.DataFrame({"Year": years[:len(nums)], "Value": [float(n.replace(',','')) for n in nums[:len(years)]]}).sort_values('Year')
                fig = px.bar(df, x="Year", y="Value", title="Trend Analysis", template="plotly_dark")
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#aeeeff"))
                return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        except: pass
        return None

    def execute(self, topic, region, mode):
        context = self._hunter(topic, region)
        
        prompt = f"""
        Act as NewsWeave Intelligence. Generate a professional executive report suitable for PDF export.
        TOPIC: {topic} | REGION: {region}
        CONTEXT: {context}
        
        FORMAT (HTML):
        <div class='pdf-header'>
            <h1>Executive Intelligence Report</h1>
            <p class='meta'>Generated by NewsWeave AI | {datetime.now().strftime('%Y-%m-%d')}</p>
        </div>
        <div class='pdf-body'>
            <h3>Executive Summary</h3>
            <p>[Concise summary of facts]</p>
            <h3>Key Findings</h3>
            <ul>[Bullet points with metrics]</ul>
            <h3>Strategic Outlook</h3>
            <p>[Forward-looking analysis]</p>
        </div>
        """
        try: report = self.llm.invoke(prompt).content if self.llm else "AI Offline. Configure API Key."
        except Exception as e: report = str(e)

        chart = self._analyst(report)
        
        # Images (Async fetch simulation for report context)
        images = []
        try:
            with DDGS() as ddgs:
                res = list(ddgs.images(f"{topic} news", max_results=4))
                for r in res: images.append({"src": r['image'], "title": r['title']})
        except: pass

        return DISCLAIMER_HTML + report, images, chart

agent = SwarmCommander()
semaphore = asyncio.Semaphore(3)
executor = ThreadPoolExecutor(max_workers=4)

# --- ROUTES ---

@app.get("/")
async def serve_index(request: Request, db: Session = Depends(get_db)):
    if db:
        stats = get_or_create_stats(db)
        count = stats.total_likes
    else:
        count = 0
    return templates.TemplateResponse("index.html", {"request": request, "regions": REGION_MAP, "total_likes": count})

@app.post("/analyze")
async def analyze(request: SearchRequest, db: Session = Depends(get_db)):
    if db:
        stats = get_or_create_stats(db)
        stats.total_prompts += 1
        stats.prompts_today += 1
        db.commit()
    
    report, images, chart = agent.execute(request.topic, request.region, request.mode)
    return JSONResponse({"report": report, "images": images, "chart": chart, "logs": []})

@app.post("/like")
async def like(db: Session = Depends(get_db)):
    count = 0
    if db:
        stats = get_or_create_stats(db)
        stats.total_likes += 1
        db.commit()
        db.refresh(stats)
        count = stats.total_likes
    return JSONResponse({"new_count": count})

# Smart Trending Image Fetcher
async def fetch_img(title):
    async with semaphore:
        loop = asyncio.get_event_loop()
        def search():
            try:
                with DDGS() as ddgs:
                    res = list(ddgs.images(f"{title} news", max_results=1))
                    return res[0]['image'] if res else None
            except: return None
        return await loop.run_in_executor(executor, search)

@app.post("/trending")
async def trending(request: TrendingRequest):
    if request.region in TRENDING_CACHE: return JSONResponse({"headlines": TRENDING_CACHE[request.region]})
    
    headlines = []
    try:
        loop = asyncio.get_event_loop()
        def get_news():
            reg = "wt-wt"
            if request.region == "India": reg = "in-en" 
            elif request.region == "USA": reg = "us-en"
            
            with DDGS() as ddgs: return list(ddgs.news(f"top news {request.region}", region=reg, max_results=8))
        
        raw_news = await loop.run_in_executor(executor, get_news)
        tasks = []
        
        for r in raw_news:
            item = {"title": r['title'], "source": r['source'], "date": r['date'], "image": r.get('image')}
            headlines.append(item)
            if not item['image']: tasks.append(fetch_img(r['title']))
            else: tasks.append(asyncio.sleep(0, result=item['image']))
            
        imgs = await asyncio.gather(*tasks)
        for i, url in enumerate(imgs):
            if not headlines[i]['image']: 
                # Category Fallback
                t = headlines[i]['title'].lower()
                if any(x in t for x in ["tech","ai","chip"]): fallback = CATEGORY_IMAGES['tech']
                elif any(x in t for x in ["market","bank","stock"]): fallback = CATEGORY_IMAGES['finance']
                elif any(x in t for x in ["war","army"]): fallback = CATEGORY_IMAGES['war']
                else: fallback = CATEGORY_IMAGES['general']
                headlines[i]['image'] = url or fallback
        
        TRENDING_CACHE[request.region] = headlines
    except Exception as e:
        logger.error(f"Trending Error: {e}")
        
    return JSONResponse({"headlines": headlines})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
