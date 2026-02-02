import os
import json
import uvicorn
import re
import time
import logging
import asyncio
import random
import httpx 
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

# --- CONFIGURATION & LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NewsWeave-Supreme")

# --- SMART KEEP-ALIVE STATE ---
# This tracks the last time a HUMAN did something.
LAST_ACTIVITY_TIMESTAMP = time.time()
# Render provides this env var automatically; fallback to a placeholder
APP_URL = os.environ.get("RENDER_EXTERNAL_URL") or "http://localhost:8000"

# --- UI CONSTANTS ---
DISCLAIMER_HTML = """
<div style="background: rgba(0, 243, 255, 0.05); border-left: 3px solid #00f3ff; padding: 12px; margin-bottom: 20px; border-radius: 4px; font-size: 0.85rem; color: #aeeeff; display: flex; align-items: center; gap: 10px; box-shadow: 0 0 15px rgba(0, 243, 255, 0.1);">
    <i class="fas fa-shield-alt"></i>
    <div><strong>AI FORENSIC REPORT:</strong> Data generated autonomously via Swarm Intelligence. Verify critical intelligence.</div>
</div>
"""

INTERNAL_API_KEY = os.environ.get("GROQ_API_KEY")

app = FastAPI()
os.makedirs("static", exist_ok=True)
os.makedirs("data", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- DATA PERSISTENCE ---
DATA_FILE = "data/stats.json"
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

# ==========================================
# 🛡️ SMART SELF-PING LOGIC (ANTI-SLEEP)
# ==========================================

def update_user_activity():
    """Call this in every route to prove a human is using the app."""
    global LAST_ACTIVITY_TIMESTAMP
    LAST_ACTIVITY_TIMESTAMP = time.time()

async def self_ping_loop():
    """The 'Ghost' user that keeps the app awake only when real users are gone."""
    # If no activity for 10 minutes, start pinging every 2 minutes.
    IDLE_LIMIT = 600 
    
    while True:
        await asyncio.sleep(120) # Check every 2 minutes
        idle_time = time.time() - LAST_ACTIVITY_TIMESTAMP
        
        if idle_time > IDLE_LIMIT:
            logger.info(f"System Idle for {int(idle_time)}s. Initiating Wake-Lock Ping...")
            try:
                async with httpx.AsyncClient() as client:
                    # We ping the root URL to trigger the server
                    response = await client.get(APP_URL, timeout=10.0)
                    logger.info(f"Wake-Lock Success: Status {response.status_code}")
            except Exception as e:
                logger.error(f"Wake-Lock Failed: {e}")

@app.on_event("startup")
async def on_startup():
    # Launches the background heartbeat
    asyncio.create_task(self_ping_loop())

# ==========================================
# 🧠 SWARM INTELLIGENCE CORE
# ==========================================

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

class SearchRequest(BaseModel):
    topic: str
    region: str
    mode: str

class TrendingRequest(BaseModel):
    region: str

class SwarmCommander:
    def __init__(self):
        self.llm = ChatGroq(temperature=0.1, model_name="llama-3.3-70b-versatile", api_key=INTERNAL_API_KEY) if INTERNAL_API_KEY else None
        self.logs = []

    def log_action(self, agent, message):
        t = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"<span style='opacity:0.6'>[{t}]</span> <b style='color:#00f3ff'>[{agent}]</b> {message}")

    def hunter_gather(self, topic, region_code, mode):
        self.log_action("HUNTER", f"Executing search protocols for: {topic} in {region_code}")
        vault = ""
        try:
            with DDGS() as ddgs:
                # Multi-stage search strategy
                q1 = f"{topic} news {datetime.now().year}"
                res = list(ddgs.text(q1, region=region_code, max_results=6))
                for r in res: vault += f"SOURCE: {r['title']}\nDATA: {r['body']}\n\n"
        except Exception as e:
            self.log_action("ERROR", f"Hunter failed: {str(e)}")
        return vault

    def vision_scan(self, topic, region_code):
        self.log_action("VISION", "Scanning for visual intelligence assets...")
        gallery = []
        try:
            with DDGS() as ddgs:
                res = list(ddgs.images(f"{topic} official", region=region_code, max_results=12))
                for r in res: gallery.append({"src": r['image'], "title": r['title']})
        except: pass
        return gallery

    def analyze_data(self, topic, context, mode):
        self.log_action("ANALYST", "Processing data through Neural Engine...")
        if not self.llm: return "SYSTEM OFFLINE: Groq Key Missing."
        
        prompt = f"""
        Role: NewsWeave Supreme Intelligence.
        Topic: {topic}
        Mode: {mode}
        Context: {context}
        Requirement: Provide a high-density, professional intelligence dossier using HTML tags. 
        Include sections for 'Executive Summary', 'Key Evidence', and 'Future Projections'.
        """
        return self.llm.invoke(prompt).content

    def mission_control(self, topic, region, mode):
        self.logs = []
        r_code = REGION_MAP.get(region, "wt-wt")
        
        raw_intel = self.hunter_gather(topic, r_code, mode)
        images = self.vision_scan(topic, r_code)
        report = self.analyze_data(topic, raw_intel, mode)
        
        return DISCLAIMER_HTML + report, images, self.logs

commander = SwarmCommander()
executor = ThreadPoolExecutor(max_workers=5)
sem = asyncio.Semaphore(3)

# ==========================================
# 🛣️ API ENDPOINTS
# ==========================================

@app.get("/")
async def index(request: Request):
    update_user_activity() # Human interaction detected
    stats = load_stats()
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "regions": REGION_MAP, 
        "total_likes": stats["total_likes"]
    })

@app.post("/analyze")
async def analyze(request: SearchRequest):
    update_user_activity() # Human interaction detected
    report, imgs, logs = commander.mission_control(request.topic, request.region, request.mode)
    return JSONResponse({"report": report, "images": imgs, "logs": logs})

@app.post("/trending")
async def trending(request: TrendingRequest):
    update_user_activity() # Human interaction detected
    # (Trending Logic Restored to Full Strength)
    region_code = REGION_MAP.get(request.region, "wt-wt")
    headlines = []
    try:
        with DDGS() as ddgs:
            news_items = list(ddgs.news("top headlines", region=region_code, max_results=8))
            for n in news_items:
                headlines.append({
                    "title": n['title'], "source": n['source'], "date": n['date'],
                    "image": n.get('image') or "https://via.placeholder.com/400x200/000/00f3ff?text=Intel"
                })
    except: pass
    return JSONResponse({"headlines": headlines})

@app.post("/like")
async def like():
    update_user_activity() # Human interaction detected
    s = load_stats()
    s["total_likes"] += 1
    save_stats(s)
    return JSONResponse({"new_count": s["total_likes"]})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
