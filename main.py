import os
import json
import uvicorn
import re
import time
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date
from fastapi import FastAPI, Request
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

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NewsWeave-Supreme")

# CYBERPUNK DISCLAIMER (Injects into HTML)
DISCLAIMER_HTML = """
<div style="background: rgba(0, 243, 255, 0.05); border-left: 3px solid #00f3ff; padding: 15px; margin-bottom: 25px; border-radius: 4px; font-size: 0.9rem; color: #aeeeff; display: flex; align-items: center; gap: 12px; box-shadow: 0 0 15px rgba(0, 243, 255, 0.1);">
    <i class="fas fa-shield-alt" style="font-size:1.2rem;"></i>
    <div><strong>FORENSIC AI REPORT:</strong> Data generated autonomously. Verify critical intelligence with primary sources.</div>
</div>
"""

INTERNAL_API_KEY = os.environ.get("GROQ_API_KEY")

app = FastAPI()
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- DATA PERSISTENCE ---
DATA_FILE = "data/stats.json"
os.makedirs("data", exist_ok=True)
TRENDING_CACHE = {} 

def load_stats():
    default = {"prompts_today": 0, "total_prompts": 0, "total_likes": 0, "date": str(date.today())}
    if not os.path.exists(DATA_FILE): return default
    try:
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
            if data.get("date") != str(date.today()):
                data["prompts_today"] = 0
                data["date"] = str(date.today())
                save_stats(data)
            return data
    except: return default

def save_stats(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)

# --- 67+ COUNTRY MAP ---
REGION_MAP = {
    "Global": "wt-wt", "Argentina": "ar-es", "Australia": "au-en", "Austria": "at-de",
    "Belgium (FR)": "be-fr", "Belgium (NL)": "be-nl", "Brazil": "br-pt", "Bulgaria": "bg-bg",
    "Canada (EN)": "ca-en", "Canada (FR)": "ca-fr", "Chile": "cl-es", "China": "cn-zh",
    "Colombia": "co-es", "Croatia": "hr-hr", "Czech Republic": "cz-cs", "Denmark": "dk-da",
    "Egypt": "xa-ar", "Estonia": "ee-et", "Finland": "fi-fi", "France": "fr-fr",
    "Germany": "de-de", "Greece": "gr-el", "Hong Kong": "hk-tzh", "Hungary": "hu-hu",
    "India": "in-en", "Indonesia": "id-en", "Ireland": "ie-en", "Israel": "il-en",
    "Italy": "it-it", "Japan": "jp-jp", "Korea": "kr-kr", "Latvia": "lv-lv",
    "Lithuania": "lt-lt", "Malaysia": "my-en", "Mexico": "mx-es", "Netherlands": "nl-nl",
    "New Zealand": "nz-en", "Norway": "no-no", "Pakistan": "pk-en", "Peru": "pe-es",
    "Philippines": "ph-en", "Poland": "pl-pl", "Portugal": "pt-pt", "Romania": "ro-ro",
    "Russia": "ru-ru", "Saudi Arabia": "xa-ar", "Singapore": "sg-en", "Slovakia": "sk-sk",
    "Slovenia": "sl-sl", "South Africa": "za-en", "Spain": "es-es", "Sweden": "se-sv",
    "Switzerland (DE)": "ch-de", "Switzerland (FR)": "ch-fr", "Taiwan": "tw-tzh",
    "Thailand": "th-th", "Turkey": "tr-tr", "UK": "uk-en", "Ukraine": "ua-uk",
    "USA": "us-en", "Vietnam": "vn-vi"
}

# --- SMART FALLBACK CATEGORIES ---
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

# ==========================================
# 🧠 SWARM INTELLIGENCE CORE
# ==========================================

class SwarmCommander:
    def __init__(self):
        self.llm = ChatGroq(temperature=0.0, model_name="llama-3.3-70b-versatile", api_key=INTERNAL_API_KEY) if INTERNAL_API_KEY else None
        self.logs = []

    def log(self, agent, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"<span style='opacity:0.6'>[{timestamp}]</span> <b style='color:#00f3ff'>[{agent}]</b> {message}"
        self.logs.append(entry)

    def _hunter_agent(self, topic, region_code, mode):
        self.log("HUNTER", f"Initiating multi-vector sweep for: {topic}")
        strategies = []
        if mode == "Fact Check": strategies = [f"{topic} verified fact check", f"is {topic} true"]
        elif mode == "Market Analysis": strategies = [f"{topic} revenue data {datetime.now().year}", f"{topic} market report"]
        elif mode == "Catalog": strategies = [f"list of {topic}", f"types of {topic}"]
        else: strategies = [f"{topic} news analysis", f"{topic} details"]
        
        vault = ""
        with DDGS() as ddgs:
            for q in strategies:
                try:
                    time.sleep(0.2) 
                    results = list(ddgs.text(q, region=region_code, max_results=4))
                    for r in results: vault += f"SOURCE: {r['title']}\nLINK: {r['href']}\nDATA: {r['body']}\n\n"
                except: pass

        if len(vault) < 200:
            try:
                page = wikipedia.summary(topic, sentences=5)
                vault += f"ENCYCLOPEDIA: {page}\n\n"
            except: pass
        return vault

    def _vision_agent(self, topic, region_code):
        gallery = []
        try:
            with DDGS() as ddgs:
                results = list(ddgs.images(f"{topic} real context", region=region_code, max_results=20))
                for r in results:
                    if len(gallery) >= 20: break
                    gallery.append({"src": r['image'], "title": r['title']})
        except: pass
        return gallery

    def _analyst_agent(self, text):
        try:
            years = re.findall(r'\b(20\d{2})\b', text)
            numbers = re.findall(r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\b', text)
            if len(years) > 1 and len(numbers) > 1:
                clean_years, clean_nums = [], []
                for i in range(min(len(years), len(numbers))):
                    try:
                        y = int(years[i])
                        v = float(numbers[i].replace(',', ''))
                        if 1900 < v < 2100: continue
                        clean_years.append(y); clean_nums.append(v)
                    except: pass
                if clean_years:
                    df = pd.DataFrame({"Year": clean_years, "Value": clean_nums}).sort_values(by="Year")
                    fig = px.scatter(df, x="Year", y="Value", title="Trend Analysis", trendline="ols" if len(df)>3 else None, template="plotly_dark")
                    fig.update_layout(xaxis=dict(dtick=1), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        except: pass
        return None

    def execute_mission(self, topic, region, mode):
        self.logs = []
        region_code = REGION_MAP.get(region, "wt-wt")
        
        context = self._hunter_agent(topic, region_code, mode)
        images = self._vision_agent(topic, region_code)
        
        if not context and not images:
             return f"{DISCLAIMER_HTML}<h3>⚠️ Mission Failed</h3><p>Data Void. Try a broader topic.</p>", [], None, self.logs

        # PDF-Optimized HTML Structure
        prompt = f"""
        You are NewsWeave Supreme. TOPIC: {topic} | REGION: {region}
        CONTEXT: {context}
        INSTRUCTIONS:
        1. Write a professional report with HTML tags (h3, p, ul).
        2. Use citations [1], [2].
        3. Structure: Executive Summary -> Key Findings -> Deep Analysis.
        """
        try: report = self.llm.invoke(prompt).content if self.llm else "LLM Offline."
        except Exception as e: report = str(e)

        final_report = DISCLAIMER_HTML + report
        chart = self._analyst_agent(final_report)
        
        return final_report, images, chart, self.logs

agent = SwarmCommander()

# ==========================================
# ⚡ ZERO-LAG IMAGE ENGINE
# ==========================================

# Semaphore limits concurrent requests to 3. This solves the "Lag" issue.
semaphore = asyncio.Semaphore(3)
executor = ThreadPoolExecutor(max_workers=4)

async def safe_fetch_image(title):
    """
    Fetches image securely. Returns Category Fallback if fails.
    """
    async with semaphore:
        loop = asyncio.get_event_loop()
        def search():
            try:
                with DDGS() as ddgs:
                    # Quick fetch (max 1)
                    imgs = list(ddgs.images(f"{title} news", max_results=1))
                    return imgs[0]['image'] if imgs else None
            except: return None
        
        img_url = await loop.run_in_executor(executor, search)
        
        # SMART FALLBACK
        if not img_url:
            t = title.lower()
            if any(x in t for x in ['ai', 'tech', 'cyber', 'data', 'chip', 'soft']): return CATEGORY_IMAGES['tech']
            elif any(x in t for x in ['market', 'stock', 'bank', 'econ', 'rate']): return CATEGORY_IMAGES['finance']
            elif any(x in t for x in ['war', 'strike', 'army', 'conflict']): return CATEGORY_IMAGES['war']
            elif any(x in t for x in ['senate', 'law', 'gov', 'trump', 'biden']): return CATEGORY_IMAGES['politics']
            else: return CATEGORY_IMAGES['general']
            
        return img_url

@app.get("/")
async def serve_interface(request: Request):
    stats = load_stats()
    return templates.TemplateResponse("index.html", {"request": request, "regions": REGION_MAP, "total_likes": stats["total_likes"]})

@app.post("/like")
async def like_endpoint():
    s = load_stats()
    s["total_likes"] += 1
    save_stats(s)
    return JSONResponse(content={"new_count": s["total_likes"]})

@app.post("/trending")
async def get_trending(request: TrendingRequest):
    # Check Cache
    if request.region in TRENDING_CACHE:
        return JSONResponse(content={"headlines": TRENDING_CACHE[request.region]})

    region_code = REGION_MAP.get(request.region, "wt-wt")
    headlines = []
    
    try:
        # 1. Fetch Text
        query = "top news" if request.region == "Global" else f"top news in {request.region}"
        loop = asyncio.get_event_loop()
        def fetch_text():
            with DDGS() as ddgs:
                return list(ddgs.news(query, region=region_code, max_results=8))
        
        results = await loop.run_in_executor(executor, fetch_text)
        
        # 2. Gather Images
        tasks = []
        for r in results:
            item = {
                "title": r['title'], "url": r['url'], "source": r['source'], "date": r['date'], "image": r.get('image', None)
            }
            headlines.append(item)
            
            if not item['image']:
                tasks.append(safe_fetch_image(r['title']))
            else:
                tasks.append(asyncio.sleep(0, result=item['image']))
        
        fetched_images = await asyncio.gather(*tasks)
        
        # 3. Merge
        for i, img_url in enumerate(fetched_images):
            if not headlines[i]['image']:
                headlines[i]['image'] = img_url

        TRENDING_CACHE[request.region] = headlines

    except Exception as e:
        logger.error(f"Trending Error: {e}")
    
    return JSONResponse(content={"headlines": headlines})

@app.post("/analyze")
async def analyze_endpoint(request: SearchRequest):
    report, images, chart, logs = agent.execute_mission(request.topic, request.region, request.mode)
    return JSONResponse(content={"report": report, "images": images, "chart": chart, "logs": logs})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
