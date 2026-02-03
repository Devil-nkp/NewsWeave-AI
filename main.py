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

# --- DATABASE ---
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

# --- CORE INTELLIGENCE AGENT ---
class SwarmCommander:
    def __init__(self):
        self.llm = ChatGroq(temperature=0.3, model_name="llama-3.3-70b-versatile", api_key=INTERNAL_API_KEY) if INTERNAL_API_KEY else None

    def _resolve_mode(self, topic, mode):
        """Auto-Detect Mode if User Selects 'Auto'"""
        if mode != "Auto": return mode
        
        t = topic.lower()
        if any(x in t for x in ['list', 'types', 'catalog', 'all']): return "Catalog"
        if any(x in t for x in ['true', 'fake', 'real', 'hoax', 'fact']): return "Fact Check"
        if any(x in t for x in ['price', 'market', 'share', 'growth', 'vs', 'compare']): return "Market Analysis"
        return "Deep Research" # Default

    def _hunter_agent(self, topic, region, mode):
        reg = "wt-wt"
        if region == "India": reg = "in-en"
        elif region == "USA": reg = "us-en"
        
        # POLYMORPHIC SEARCH STRATEGY
        queries = []
        if mode == "Catalog":
            queries = [f"list of {topic}", f"types of {topic}", f"{topic} comprehensive list"]
        elif mode == "Fact Check":
            queries = [f"is {topic} true", f"{topic} fact check", f"{topic} hoax debunk"]
        elif mode == "Market Analysis":
            queries = [f"{topic} market share {datetime.now().year}", f"{topic} revenue statistics", f"{topic} growth rate"]
        else: # Deep Research
            queries = [f"{topic} news analysis", f"{topic} detailed report", f"{topic} implications"]

        vault = ""
        try:
            with DDGS() as ddgs:
                for q in queries:
                    # Fetching more results per query for depth
                    results = list(ddgs.text(q, region=reg, max_results=5))
                    for r in results:
                        vault += f"SOURCE: {r['title']}\nLINK: {r['href']}\nCONTENT: {r['body']}\n\n"
                    time.sleep(0.1) # Respectful delay
        except: pass
        
        if len(vault) < 500:
            try: vault += f"WIKI-SUMMARY: {wikipedia.summary(topic, sentences=8)}"
            except: pass
            
        return vault if vault else "No specific data found via Live Search."

    def _vision_agent(self, topic):
        gallery = []
        try:
            with DDGS() as ddgs:
                # MASSIVE SEARCH: Fetch 40 images to ensure 25 good ones
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
                        clean_years.append(y)
                        clean_nums.append(v)
                    except: pass
                
                if clean_years:
                    df = pd.DataFrame({"Year": clean_years, "Metric": clean_nums}).sort_values('Year')
                    df = df.groupby('Year', as_index=False).mean()
                    
                    fig = px.area(df, x="Year", y="Metric", title=f"Trend Analysis", template="plotly_dark", markers=True)
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)', 
                        plot_bgcolor='rgba(0,0,0,0)', 
                        font=dict(color="#aeeeff"),
                        autosize=True
                    )
                    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        except Exception as e:
            logger.error(f"Chart Error: {e}")
        return None

    def execute(self, topic, region, mode):
        # 1. Resolve Mode
        active_mode = self._resolve_mode(topic, mode)
        
        # 2. Hunt
        context = self._hunter_agent(topic, region, active_mode)
        
        # 3. Vision
        images = self._vision_agent(topic)
        
        # 4. Synthesize (Polymorphic Prompting)
        system_prompt = f"You are NewsWeave Supreme. MODE: {active_mode.upper()}."
        
        if active_mode == "Catalog":
            structure = """
            - <h2>COMPREHENSIVE CATALOG</h2>
            - <h2>LIST OF ITEMS</h2> (Use <ul><li> format strictly)
            - <h2>DETAILED SPECS/DESCRIPTIONS</h2>
            - <h2>SOURCES</h2>
            """
        elif active_mode == "Fact Check":
            structure = """
            - <h2>VERDICT: [TRUE / FALSE / UNVERIFIED]</h2>
            - <h2>EVIDENCE ANALYSIS</h2>
            - <h2>ORIGIN OF CLAIM</h2>
            - <h2>CONCLUSION</h2>
            """
        elif active_mode == "Market Analysis":
            structure = """
            - <h2>MARKET EXECUTIVE SUMMARY</h2>
            - <h2>KEY FINANCIAL METRICS</h2> (Use bullet points with numbers)
            - <h2>COMPETITIVE LANDSCAPE (Comparisons)</h2>
            - <h2>FUTURE FORECAST</h2>
            """
        else: # Deep Research
            structure = """
            - <h2>EXECUTIVE SUMMARY</h2>
            - <h2>KEY FINDINGS & METRICS</h2>
            - <h2>DEEP DIVE ANALYSIS</h2> (Long form, multiple paragraphs)
            - <h2>GEOPOLITICAL/MARKET IMPACT</h2>
            - <h2>SOURCES</h2>
            """

        prompt = f"""
        {system_prompt}
        TOPIC: {topic} | REGION: {region}
        CONTEXT: {context}
        
        INSTRUCTIONS:
        1. Write a LONG, ACCURATE report based ONLY on context.
        2. Use this structure: {structure}
        3. Use HTML tags (h2, p, ul, li, strong). NO Markdown.
        """
        
        try: report = self.llm.invoke(prompt).content if self.llm else "LLM Offline."
        except Exception as e: report = f"Analysis Interrupted: {str(e)}"

        # 5. Charting
        chart = self._analyst_agent(report + context)
        
        final_html = DISCLAIMER_HTML + report
        return final_html, images, chart

agent = SwarmCommander()
semaphore = asyncio.Semaphore(4)
executor = ThreadPoolExecutor(max_workers=6)

# --- ENDPOINTS ---

@app.get("/")
async def index(request: Request, db: Session = Depends(get_db)):
    count = 0
    if db:
        s = get_or_create_stats(db)
        count = s.total_likes
    return templates.TemplateResponse("index.html", {"request": request, "regions": REGION_MAP, "total_likes": count})

# PROXY IMAGE FOR PDF (Crucial for PDF fix)
@app.get("/proxy-image")
async def proxy_image(url: str):
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            return Response(content=resp.content, media_type=resp.headers.get("content-type", "image/jpeg"))
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

async def fetch_img_with_timeout(title):
    async with semaphore:
        loop = asyncio.get_event_loop()
        def s():
            try:
                with DDGS() as ddgs:
                    r = list(ddgs.images(f"{title} news", max_results=1))
                    return r[0]['image'] if r else None
            except: return None
        
        try:
            return await asyncio.wait_for(loop.run_in_executor(executor, s), timeout=2.0)
        except asyncio.TimeoutError:
            return None

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
            if not h['image']: tasks.append(fetch_img_with_timeout(r['title']))
            else: tasks.append(asyncio.sleep(0, result=h['image']))
            
        imgs = await asyncio.gather(*tasks)
        for i, url in enumerate(imgs):
            if not headlines[i]['image']:
                t = headlines[i]['title'].lower()
                if "tech" in t or "ai" in t: fallback = CATEGORY_IMAGES['tech']
                elif "market" in t or "bank" in t: fallback = CATEGORY_IMAGES['finance']
                elif "war" in t: fallback = CATEGORY_IMAGES['war']
                else: fallback = CATEGORY_IMAGES['general']
                headlines[i]['image'] = url or fallback
        
        TRENDING_CACHE[request.region] = headlines
        return JSONResponse({"headlines": headlines})
    except:
        return JSONResponse({"headlines": []})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
