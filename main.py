import os
import json
import uvicorn
import re
import time
import logging
import asyncio
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

# SAFETY DISCLAIMER
DISCLAIMER_HTML = """
<div style="background: rgba(255, 165, 0, 0.1); border-left: 3px solid #ffaa00; padding: 12px; margin-bottom: 20px; border-radius: 4px; font-size: 0.85rem; color: #ffcc80; display: flex; align-items: center; gap: 10px;">
    <i class="fas fa-shield-alt"></i>
    <div><strong>AI GENERATED CONTENT:</strong> Verify critical data. This report is for educational research purposes.</div>
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

def load_stats():
    default = {"prompts_today": 0, "total_prompts": 0, "total_likes": 2450, "date": str(date.today())}
    if not os.path.exists(DATA_FILE): return default
    try:
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
            return data
    except: return default

def save_stats(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)

# --- ULTIMATE GLOBAL REGION MAP (65+ COUNTRIES) ---
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

# ==========================================
# 🧠 SWARM INTELLIGENCE CORE (v35)
# ==========================================

class SwarmCommander:
    def __init__(self):
        if not INTERNAL_API_KEY:
             self.llm = None
        else:
             self.llm = ChatGroq(temperature=0.0, model_name="llama-3.3-70b-versatile", api_key=INTERNAL_API_KEY)
        self.logs = []

    def log(self, agent, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"<span style='opacity:0.6'>[{timestamp}]</span> <b style='color:#00f3ff'>[{agent}]</b> {message}"
        self.logs.append(entry)

    def _hunter_agent(self, topic, region_code, mode):
        self.log("HUNTER", f"Initiating multi-vector sweep for: {topic}")
        strategies = []
        if mode == "Fact Check": strategies = [f"{topic} fact check", f"is {topic} true"]
        elif mode == "Market Analysis": strategies = [f"{topic} revenue data {datetime.now().year}", f"{topic} market report"]
        elif mode == "Catalog": strategies = [f"list of {topic}", f"types of {topic}"]
        else: strategies = [f"{topic} news", f"{topic} analysis"]
        
        vault = ""
        with DDGS() as ddgs:
            for q in strategies:
                try:
                    time.sleep(0.5) 
                    results = list(ddgs.text(q, region=region_code, max_results=4))
                    for r in results: vault += f"SOURCE: {r['title']}\nLINK: {r['href']}\nDATA: {r['body']}\n\n"
                except: pass

        if len(vault) < 500:
            try:
                page = wikipedia.summary(topic, sentences=5)
                vault += f"ENCYCLOPEDIA: {page}\n\n"
            except: pass
        return vault

    def _vision_agent(self, topic, region_code):
        gallery = []
        try:
            with DDGS() as ddgs:
                results = list(ddgs.images(f"{topic} real photo", region=region_code, max_results=20))
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
        
        # Hunter
        context = self._hunter_agent(topic, region_code, mode)
        images = self._vision_agent(topic, region_code)
        
        if not context and not images:
             return f"{DISCLAIMER_HTML}<h3>⚠️ Mission Failed</h3><p>Data Void.</p>", [], None, self.logs

        # Editor
        prompt = f"""
        You are NewsWeave Supreme. TOPIC: {topic} | REGION: {region}
        CONTEXT: {context}
        INSTRUCTIONS:
        1. Write a structured report with HTML tags (h3, p, ul).
        2. Use citations [1], [2].
        3. If analyzing markets, extract numbers.
        """
        try: report = self.llm.invoke(prompt).content if self.llm else "LLM Offline."
        except Exception as e: report = str(e)

        final_report = DISCLAIMER_HTML + report
        chart = self._analyst_agent(final_report)
        
        return final_report, images, chart, self.logs

agent = SwarmCommander()

# ==========================================
# 🌐 API ROUTES
# ==========================================

@app.get("/")
async def serve_interface(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "regions": REGION_MAP})

@app.post("/trending")
async def get_trending(request: TrendingRequest):
    region_code = REGION_MAP.get(request.region, "wt-wt")
    headlines = []
    try:
        with DDGS() as ddgs:
            # Query Logic: Customizes the trending search based on region
            query = "top news stories" if request.region == "Global" else f"top news in {request.region}"
            results = list(ddgs.news(query, region=region_code, max_results=8))
            
            for r in results:
                # Fallback for missing images in DDGS response
                img = r.get('image', None)
                if not img: img = "https://via.placeholder.com/300x150/000000/00f3ff?text=NewsWeave+Intel"
                
                headlines.append({
                    "title": r['title'],
                    "url": r['url'],
                    "source": r['source'],
                    "date": r['date'],
                    "image": img
                })
    except Exception as e:
        logger.error(f"Trending Error: {e}")
    
    return JSONResponse(content={"headlines": headlines})

@app.post("/analyze")
async def analyze_endpoint(request: SearchRequest):
    report, images, chart, logs = agent.execute_mission(request.topic, request.region, request.mode)
    return JSONResponse(content={"report": report, "images": images, "chart": chart, "logs": logs})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
