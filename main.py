import os
import json
import re
import time
import logging
import asyncio
import httpx
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date
from fastapi import FastAPI, Request, Response
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
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NewsWeave-Supreme")

# API KEYS (Set these in Vercel Settings -> Environment Variables)
INTERNAL_API_KEY = os.environ.get("GROQ_API_KEY") 
MONGODB_URI = os.environ.get("MONGODB_URI")

# --- MONGODB ATLAS CONNECTION (Serverless Optimized) ---
class AtlasClient:
    def __init__(self, uri):
        self.client = None
        self.db = None
        self.uri = uri

    def connect(self):
        if self.client: return # Already connected
        if self.uri:
            try:
                # connectTimeoutMS is vital for serverless to fail fast if needed
                self.client = MongoClient(self.uri, serverSelectionTimeoutMS=3000, connectTimeoutMS=3000)
                self.client.admin.command('ping')
                self.db = self.client["newsweave_db"]
                logger.info("✅ Connected to Atlas")
            except Exception as e:
                logger.error(f"❌ Connection Failed: {e}")
                self.client = None

    def get_stats(self):
        self.connect()
        if self.db is None: return {"total_likes": 0, "total_prompts": 0}
        
        try:
            stats = self.db.global_stats.find_one({"_id": "metrics"})
            if not stats:
                initial_data = {"_id": "metrics", "total_likes": 0, "total_prompts": 0, "prompts_today": 0, "last_date": str(date.today())}
                self.db.global_stats.insert_one(initial_data)
                return initial_data
                
            if stats.get("last_date") != str(date.today()):
                self.db.global_stats.update_one({"_id": "metrics"}, {"$set": {"prompts_today": 0, "last_date": str(date.today())}})
                stats["prompts_today"] = 0
            return stats
        except: return {"total_likes": 0}

    def increment_prompt(self):
        self.connect()
        if self.db is not None:
            try: self.db.global_stats.update_one({"_id": "metrics"}, {"$inc": {"total_prompts": 1, "prompts_today": 1}})
            except: pass

    def increment_like(self):
        self.connect()
        if self.db is not None:
            try:
                self.db.global_stats.find_one_and_update(
                    {"_id": "metrics"},
                    {"$inc": {"total_likes": 1}},
                    upsert=True
                )
                return self.get_stats()["total_likes"]
            except: pass
        return 0

mongo = AtlasClient(MONGODB_URI)

# --- APP SETUP ---
app = FastAPI()

# Mount Static & Templates
# Note: In Vercel, static files are read-only.
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

TRENDING_CACHE = {} 
CATEGORY_IMAGES = {
    "tech": "https://images.unsplash.com/photo-1518770660439-4636190af475?w=600&q=80",
    "finance": "https://images.unsplash.com/photo-1611974765270-ca1258634369?w=600&q=80",
    "war": "https://images.unsplash.com/photo-1557426272-fc759fdf7a8d?w=600&q=80",
    "politics": "https://images.unsplash.com/photo-1555848962-6e79363ec58f?w=600&q=80",
    "health": "https://images.unsplash.com/photo-1532938911079-1b06ac7ceec7?w=600&q=80",
    "general": "https://images.unsplash.com/photo-1504711434969-e33886168f5c?w=600&q=80"
}

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

class SearchRequest(BaseModel):
    topic: str
    region: str
    mode: str

class TrendingRequest(BaseModel):
    region: str

# --- CORE INTELLIGENCE AGENT ---
class SwarmCommander:
    def __init__(self):
        self.llm = ChatGroq(temperature=0.1, model_name="llama-3.3-70b-versatile", api_key=INTERNAL_API_KEY) if INTERNAL_API_KEY else None

    def _resolve_mode(self, topic, mode):
        if mode != "Auto": return mode
        t = topic.lower()
        if any(x in t for x in ['list', 'types', 'catalog', 'all']): return "Catalog"
        if any(x in t for x in ['true', 'fake', 'real', 'hoax', 'fact']): return "Fact Check"
        if any(x in t for x in ['price', 'market', 'share', 'growth', 'vs', 'compare']): return "Market Analysis"
        return "Deep Research"

    def _hunter_agent(self, topic, region, mode):
        reg = "wt-wt"
        if region == "India": reg = "in-en"
        elif region == "USA": reg = "us-en"
        
        queries = []
        current_year = datetime.now().year
        if mode == "Catalog":
            queries = [f"comprehensive list of {topic}", f"types of {topic} details"]
        elif mode == "Fact Check":
            queries = [f"is {topic} true or false", f"{topic} official fact check", f"{topic} credible news"]
        elif mode == "Market Analysis":
            queries = [f"{topic} market size {current_year}", f"{topic} revenue growth statistics", f"{topic} financial report"]
        else:
            queries = [f"{topic} latest comprehensive report", f"{topic} in-depth analysis", f"{topic} key statistics"]

        vault = ""
        try:
            with DDGS() as ddgs:
                # Reduced max_results to prevent Vercel timeouts (10s limit on free tier)
                for q in queries:
                    results = list(ddgs.text(q, region=reg, max_results=3)) 
                    for r in results:
                        vault += f"SOURCE: {r['title']}\nLINK: {r['href']}\nCONTENT: {r['body']}\n\n"
                    # Removed time.sleep to speed up execution
        except: pass
        
        if len(vault) < 200:
            try: vault += f"WIKI-SUMMARY: {wikipedia.summary(topic, sentences=6)}"
            except: pass
            
        return vault if vault else "Insufficient Data."

    def _vision_agent(self, topic):
        gallery = []
        try:
            with DDGS() as ddgs:
                results = list(ddgs.images(f"{topic} news high quality", region="wt-wt", max_results=15))
                for r in results:
                    if len(gallery) >= 15: break
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
                        if 1990 <= y <= 2035:
                            clean_years.append(y)
                            clean_nums.append(v)
                    except: pass
                
                if clean_years:
                    df = pd.DataFrame({"Year": clean_years, "Metric": clean_nums}).sort_values('Year')
                    df = df.groupby('Year', as_index=False).mean()
                    
                    fig = px.area(df, x="Year", y="Metric", title=f"Trend Analysis Detected", template="plotly_dark", markers=True)
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)', 
                        plot_bgcolor='rgba(0,0,0,0)', 
                        font=dict(color="#aeeeff"),
                        autosize=True,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        except: pass
        return None

    def execute(self, topic, region, mode):
        active_mode = self._resolve_mode(topic, mode)
        context = self._hunter_agent(topic, region, active_mode)
        images = self._vision_agent(topic)
        
        system_prompt = f"You are NewsWeave Supreme. MODE: {active_mode.upper()}."
        
        if active_mode == "Catalog": structure = "- <h2>CATALOG OVERVIEW</h2>\n- <h2>DETAILED INVENTORY</h2> (Use <ul><li>)\n- <h2>SPECIFICATIONS</h2>\n- <h2>SOURCES</h2>"
        elif active_mode == "Fact Check": structure = "- <h2>VERDICT: [TRUE / FALSE / UNVERIFIED]</h2>\n- <h2>EVIDENCE BREAKDOWN</h2>\n- <h2>ORIGIN TRACING</h2>\n- <h2>CONCLUSION</h2>"
        elif active_mode == "Market Analysis": structure = "- <h2>MARKET SNAPSHOT</h2>\n- <h2>FINANCIAL METRICS</h2> (Bulleted numbers)\n- <h2>COMPETITIVE LANDSCAPE</h2>\n- <h2>FUTURE FORECAST</h2>"
        else: structure = "- <h2>EXECUTIVE SUMMARY</h2>\n- <h2>KEY METRICS</h2>\n- <h2>DEEP DIVE ANALYSIS</h2>\n- <h2>GEOPOLITICAL IMPACT</h2>\n- <h2>SOURCES</h2>"

        prompt = f"""
        {system_prompt}
        TOPIC: {topic} | REGION: {region}
        CONTEXT: {context}
        
        INSTRUCTIONS:
        1. Write a professional report based ONLY on context.
        2. STRUCTURE IS MANDATORY: {structure}
        3. Use HTML tags (h2, p, ul, li, strong). NO Markdown.
        """
        
        try: report = self.llm.invoke(prompt).content if self.llm else "LLM Offline."
        except Exception as e: report = f"Analysis Interrupted: {str(e)}"

        chart = self._analyst_agent(report + context)
        final_html = DISCLAIMER_HTML + report
        return final_html, images, chart

agent = SwarmCommander()
semaphore = asyncio.Semaphore(5)
executor = ThreadPoolExecutor(max_workers=8)

# --- ENDPOINTS ---

@app.get("/")
async def index(request: Request):
    stats = mongo.get_stats()
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "regions": REGION_MAP, 
        "total_likes": stats.get("total_likes", 0)
    })

@app.get("/proxy-image")
async def proxy_image(url: str):
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.get(url)
            return Response(content=resp.content, media_type=resp.headers.get("content-type", "image/jpeg"))
    except:
        return Response(status_code=404)

@app.post("/analyze")
async def analyze(request: SearchRequest):
    mongo.increment_prompt()
    report, images, chart = agent.execute(request.topic, request.region, request.mode)
    return JSONResponse({"report": report, "images": images, "chart": chart})

@app.post("/like")
async def like():
    new_count = mongo.increment_like()
    return JSONResponse({"new_count": new_count})

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
        except asyncio.TimeoutError: return None

@app.post("/trending")
async def trending(request: TrendingRequest):
    if request.region in TRENDING_CACHE: 
        if (time.time() - TRENDING_CACHE[request.region]['time']) < 900:
            return JSONResponse({"headlines": TRENDING_CACHE[request.region]['data']})
    
    try:
        loop = asyncio.get_event_loop()
        def get_n():
            reg = "wt-wt"
            if request.region == "India": reg = "in-en"
            elif request.region == "USA": reg = "us-en"
            elif request.region == "UK": reg = "uk-en"
            
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
                elif "health" in t: fallback = CATEGORY_IMAGES['health']
                else: fallback = CATEGORY_IMAGES['general']
                headlines[i]['image'] = url or fallback
        
        TRENDING_CACHE[request.region] = {"data": headlines, "time": time.time()}
        return JSONResponse({"headlines": headlines})
    except:
        return JSONResponse({"headlines": []})

# Note: No uvicorn.run here. Vercel handles the server execution.
