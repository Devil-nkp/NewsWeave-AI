import os
import json
import uvicorn
import re
import time
import logging
from datetime import datetime, date
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS 
import wikipedia
from textblob import TextBlob
import pandas as pd
import plotly.express as px
import plotly.utils
from collections import Counter

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NewsWeave-Supreme")

INTERNAL_API_KEY = os.environ.get("GROQ_API_KEY", "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")

app = FastAPI()
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- PERSISTENCE DB ---
DATA_FILE = "data/stats.json"
os.makedirs("data", exist_ok=True)

def load_stats():
    default = {"prompts_today": 0, "total_prompts": 0, "total_likes": 2450, "date": str(date.today())}
    if not os.path.exists(DATA_FILE): return default
    try:
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
            if data.get("date") != str(date.today()):
                data["prompts_today"] = 0
                data["date"] = str(date.today())
            return data
    except: return default

def save_stats(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)

# --- GLOBAL REGION MAP ---
REGION_MAP = {
    "Global": "wt-wt", "USA": "us-en", "India": "in-en", "UK": "uk-en",
    "China": "cn-zh", "Japan": "jp-jp", "Germany": "de-de", "France": "fr-fr",
    "Russia": "ru-ru", "Brazil": "br-pt", "Canada": "ca-en", "Australia": "au-en",
    "Argentina": "ar-es", "Austria": "at-de", "Belgium": "be-fr", "Bulgaria": "bg-bg",
    "Chile": "cl-es", "Colombia": "co-es", "Croatia": "hr-hr", "Czech Republic": "cz-cs",
    "Denmark": "dk-da", "Egypt": "xa-ar", "Estonia": "ee-et", "Finland": "fi-fi",
    "Greece": "gr-el", "Hong Kong": "hk-tzh", "Hungary": "hu-hu", "Indonesia": "id-en",
    "Ireland": "ie-en", "Israel": "il-en", "Italy": "it-it", "Korea": "kr-kr",
    "Latvia": "lv-lv", "Lithuania": "lt-lt", "Malaysia": "my-en", "Mexico": "mx-es",
    "Netherlands": "nl-nl", "New Zealand": "nz-en", "Norway": "no-no", "Pakistan": "pk-en",
    "Peru": "pe-es", "Philippines": "ph-en", "Poland": "pl-pl", "Portugal": "pt-pt",
    "Romania": "ro-ro", "Saudi Arabia": "xa-ar", "Singapore": "sg-en", "Slovakia": "sk-sk",
    "Slovenia": "sl-sl", "South Africa": "za-en", "Spain": "es-es", "Sweden": "se-sv",
    "Switzerland": "ch-de", "Taiwan": "tw-tzh", "Thailand": "th-th", "Turkey": "tr-tr",
    "Ukraine": "ua-uk", "Vietnam": "vn-vi"
}

class SearchRequest(BaseModel):
    topic: str
    region: str
    mode: str

# ==========================================
# 🧠 SWARM INTELLIGENCE CORE (v30)
# ==========================================

class SwarmCommander:
    def __init__(self):
        self.llm = ChatGroq(temperature=0.0, model_name="llama-3.3-70b-versatile", api_key=INTERNAL_API_KEY)
        self.date_str = datetime.now().strftime("%B %d, %Y")
        self.logs = []

    def log(self, agent, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] <b>[{agent}]</b> {message}"
        self.logs.append(entry)
        print(entry)

    def _hunter_agent(self, topic, region_code, mode):
        """
        The Hunter: Responsible for retrieval.
        Uses Polymorphic Search to bypass blocks.
        """
        self.log("HUNTER", f"Initiating multi-vector sweep for: {topic}")
        strategies = [f"{topic} news", f"{topic} detailed analysis"]
        
        if mode == "Fact Check": strategies = [f"{topic} fact check", f"is {topic} true"]
        elif mode == "Market Analysis": strategies = [f"{topic} market data {datetime.now().year}", f"{topic} revenue"]
        elif mode == "Catalog": strategies = [f"list of {topic}", f"types of {topic}"]
        
        vault = ""
        try:
            with DDGS() as ddgs:
                for q in strategies:
                    # News Vector
                    try:
                        results = list(ddgs.news(q, region=region_code, max_results=5))
                        for r in results: 
                            vault += f"SOURCE: {r['title']} ({r['date']})\nLINK: {r['url']}\nDATA: {r['body']}\n\n"
                    except: pass
                    
                    # Text Vector (Fallback)
                    if len(vault) < 1000:
                        try:
                            results = list(ddgs.text(q, region=region_code, backend="lite", max_results=6))
                            for r in results: 
                                vault += f"SOURCE: {r['title']}\nLINK: {r['href']}\nDATA: {r['body']}\n\n"
                        except: pass
        except Exception as e:
            self.log("HUNTER", f"Search error: {e}")
        
        if not vault:
            self.log("HUNTER", "Web search failed. Activating Encyclopedia protocol.")
            try: vault = f"WIKIPEDIA: {wikipedia.summary(topic, sentences=8)}"
            except: pass
            
        return vault

    def _vision_agent(self, topic, region_code, context):
        """
        The Visionary: Forensically filters images.
        """
        self.log("VISION", "Scanning for forensic visual evidence...")
        gallery = []
        seen = set()
        blacklist = ["ai generated", "cartoon", "vector", "drawing", "clipart", "logo", "icon", "render"]
        
        # Context extraction
        keywords = re.findall(r'\b[A-Z][a-z]+\b', context)
        common = [word for word, _ in Counter(keywords).most_common(2)]
        query = f"{topic} {' '.join(common)} real photo"
        
        try:
            with DDGS() as ddgs:
                results = list(ddgs.images(query, region=region_code, max_results=25))
                for r in results:
                    if len(gallery) >= 50: break
                    t = r.get('title', '').lower()
                    src = r.get('image', '')
                    if src not in seen and not any(b in t for b in blacklist):
                        gallery.append({"src": src, "title": r['title']})
                        seen.add(src)
        except: pass
        
        self.log("VISION", f"Secured {len(gallery)} verified images.")
        return gallery

    def _analyst_agent(self, text):
        """
        The Analyst: Extracts data for charts.
        """
        self.log("ANALYST", "Parsing data for predictive modeling...")
        try:
            years = re.findall(r'\b(20\d{2})\b', text)
            numbers = re.findall(r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\b', text)
            
            if len(years) > 1 and len(numbers) > 1:
                clean_years, clean_nums = [], []
                for i in range(min(len(years), len(numbers))):
                    try:
                        y = int(years[i])
                        v = float(numbers[i].replace(',', ''))
                        clean_years.append(y); clean_nums.append(v)
                    except: pass
                
                if clean_years:
                    df = pd.DataFrame({"Year": clean_years, "Value": clean_nums})
                    try: 
                        fig = px.scatter(df, x="Year", y="Value", title="Trend Analysis (Projected)", trendline="ols", template="plotly_dark")
                    except:
                        fig = px.scatter(df, x="Year", y="Value", title="Data Distribution", template="plotly_dark")
                    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        except: pass
        return None

    def execute_mission(self, topic, region, mode):
        self.logs = [] # Reset logs
        self.log("COMMANDER", f"Mission Initialized: {topic} [{mode}]")
        
        # 1. Update Stats
        s = load_stats()
        s["prompts_today"] += 1; s["total_prompts"] += 1
        save_stats(s)

        # 2. Hunter Phase
        region_code = REGION_MAP.get(region, "wt-wt")
        context = self._hunter_agent(topic, region_code, mode)
        
        if not context:
            return "⚠️ Mission Failed: Data Void.", [], None, self.logs

        # 3. Editor Phase (LLM Synthesis)
        self.log("EDITOR", "Synthesizing Intelligence Dossier...")
        
        structure = ""
        if mode == "Catalog": structure = "CATALOG MODE: List EVERY item found. Bullet points."
        elif mode == "Fact Check": structure = "VERDICT MODE: State VERIFIED or DEBUNKED immediately."
        else: structure = "Structure: <h3>Executive Verdict</h3>, <h3>Deep Dive Analysis</h3>, <h3>Key Evidence</h3>, <h3>Strategic Outlook</h3>."

        prompt = f"""
        You are NewsWeave Supreme. TOPIC: {topic} | MODE: {mode} | REGION: {region}
        DATA: {context}
        INSTRUCTIONS:
        1. {structure}
        2. **Citations:** <a href='URL' target='_blank' style='color:#00c6ff'>[Source]</a>.
        3. **No Hallucinations:** If data is missing, admit it.
        4. Use HTML tags.
        """
        
        try: report = self.llm.invoke(prompt).content
        except Exception as e: report = f"AI Error: {e}"

        # 4. Analyst & Vision Phase
        images = self._vision_agent(topic, region_code, context)
        chart = self._analyst_agent(report)
        
        self.log("COMMANDER", "Mission Complete.")
        return report, images, chart, self.logs

agent = SwarmCommander()

# ==========================================
# 🌐 API ROUTES
# ==========================================

@app.get("/")
async def serve_interface(request: Request):
    stats = load_stats()
    return templates.TemplateResponse("index.html", {"request": request, "total_likes": stats["total_likes"]})

@app.post("/analyze")
async def analyze_endpoint(request: SearchRequest):
    report, images, chart, logs = agent.execute_mission(request.topic, request.region, request.mode)
    return JSONResponse(content={"report": report, "images": images, "chart": chart, "logs": logs})

@app.post("/like")
async def like_endpoint():
    s = load_stats()
    s["total_likes"] += 1
    save_stats(s)
    return JSONResponse(content={"new_count": s["total_likes"]})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
