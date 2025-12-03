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

# --- SYSTEM CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NewsWeave-v22")

# API Key Strategy
INTERNAL_API_KEY = os.environ.get("GROQ_API_KEY", "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")

app = FastAPI()

# Mount Static Files (Create a folder named 'static' if you have local videos)
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# --- PERSISTENCE DB ---
DATA_FILE = "data/stats.json"
os.makedirs("data", exist_ok=True)

def load_stats():
    default = {"prompts_today": 0, "total_prompts": 0, "total_likes": 1540, "date": str(date.today())}
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

# --- GLOBAL REGION MAP (50+ Countries) ---
REGION_MAP = {
    "Global": "wt-wt", "USA": "us-en", "India": "in-en", "UK": "uk-en",
    "Argentina": "ar-es", "Australia": "au-en", "Austria": "at-de",
    "Belgium (fr)": "be-fr", "Belgium (nl)": "be-nl", "Brazil": "br-pt",
    "Bulgaria": "bg-bg", "Canada (en)": "ca-en", "Canada (fr)": "ca-fr",
    "Chile": "cl-es", "China": "cn-zh", "Colombia": "co-es", "Croatia": "hr-hr",
    "Czech Republic": "cz-cs", "Denmark": "dk-da", "Egypt": "xa-ar", "Estonia": "ee-et",
    "Finland": "fi-fi", "France": "fr-fr", "Germany": "de-de", "Greece": "gr-el",
    "Hong Kong": "hk-tzh", "Hungary": "hu-hu", "Indonesia": "id-en",
    "Ireland": "ie-en", "Israel": "il-en", "Italy": "it-it", "Japan": "jp-jp",
    "Korea": "kr-kr", "Latvia": "lv-lv", "Lithuania": "lt-lt", "Malaysia": "my-en",
    "Mexico": "mx-es", "Netherlands": "nl-nl", "New Zealand": "nz-en",
    "Norway": "no-no", "Pakistan": "pk-en", "Peru": "pe-es", "Philippines": "ph-en",
    "Poland": "pl-pl", "Portugal": "pt-pt", "Romania": "ro-ro", "Russia": "ru-ru",
    "Saudi Arabia": "xa-ar", "Singapore": "sg-en", "Slovakia": "sk-sk",
    "Slovenia": "sl-sl", "South Africa": "za-en", "Spain": "es-es",
    "Sweden": "se-sv", "Switzerland (de)": "ch-de", "Switzerland (fr)": "ch-fr",
    "Taiwan": "tw-tzh", "Thailand": "th-th", "Turkey": "tr-tr",
    "Ukraine": "ua-uk", "Vietnam": "vn-vi"
}

class SearchRequest(BaseModel):
    topic: str
    region: str
    mode: str

# ==========================================
# 🧠 SINGULARITY AGENT CORE
# ==========================================

class SingularityAgent:
    def __init__(self):
        self.llm = ChatGroq(temperature=0.0, model_name="llama-3.3-70b-versatile", api_key=INTERNAL_API_KEY)
        self.date_str = datetime.now().strftime("%B %d, %Y")

    def _determine_mode(self, topic):
        t = topic.lower()
        if any(x in t for x in ['all', 'list', 'types', 'catalog', 'every', 'classification']):
            return "Catalog"
        if any(x in t for x in ['stock', 'price', 'market', 'growth', 'economy', 'revenue']):
            return "Market Analysis"
        if any(x in t for x in ['fake', 'real', 'true', 'hoax', 'fact', 'verify']):
            return "Fact Check"
        return "Deep Research"

    def _smart_image_sweep(self, topic, region_code):
        """Hunts for 20+ Real Images"""
        gallery = []
        seen = set()
        blacklist = ["ai generated", "cartoon", "vector", "drawing", "clipart", "logo", "icon"]
        
        queries = [f"{topic} news photo", f"{topic} real life", f"{topic} event", f"{topic} official"]
        
        try:
            with DDGS() as ddgs:
                for q in queries:
                    if len(gallery) >= 30: break
                    try:
                        results = list(ddgs.images(q, region=region_code, max_results=20))
                        for r in results:
                            if len(gallery) >= 30: break
                            t = r.get('title', '').lower()
                            src = r.get('image', '')
                            if src not in seen and not any(b in t for b in blacklist):
                                gallery.append({"src": src, "title": r['title']})
                                seen.add(src)
                    except: continue
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"Image Error: {e}")
            
        return gallery

    def _execute_polymorphic_search(self, topic, region, mode):
        region_code = REGION_MAP.get(region, "wt-wt")
        active_mode = self._determine_mode(topic) if mode == "Auto" else mode
        
        strategies = [f"{topic} latest news", f"{topic} analysis"]
        if active_mode == "Catalog":
            strategies = [f"list of all {topic}", f"types of {topic} details"]
        elif active_mode == "Fact Check":
            strategies = [f"is {topic} true", f"{topic} official fact check"]
            
        vault = ""
        try:
            with DDGS() as ddgs:
                for q in strategies:
                    # Try News
                    try:
                        news = list(ddgs.news(q, region=region_code, max_results=5))
                        for r in news: vault += f"SOURCE: {r['title']} ({r['date']})\nLINK: {r['url']}\nINFO: {r['body']}\n\n"
                    except: pass
                    
                    # Try Text (Fallback)
                    if len(vault) < 1000:
                        try:
                            text = list(ddgs.text(q, region=region_code, backend="lite", max_results=6))
                            for r in text: vault += f"SOURCE: {r['title']}\nLINK: {r['href']}\nINFO: {r['body']}\n\n"
                        except: pass
        except: pass
        
        return vault, active_mode, region_code

    def _generate_chart(self, text):
        try:
            years = re.findall(r'\b(20\d{2})\b', text)
            numbers = re.findall(r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\b', text)
            if len(years) > 2 and len(numbers) > 2:
                clean_years, clean_nums = [], []
                for i in range(min(len(years), len(numbers))):
                    try:
                        y = int(years[i])
                        v = float(numbers[i].replace(',', ''))
                        clean_years.append(y); clean_nums.append(v)
                    except: pass
                if clean_years:
                    df = pd.DataFrame({"Year": clean_years, "Value": clean_nums})
                    try: fig = px.scatter(df, x="Year", y="Value", title="Data Trend", trendline="ols", template="plotly_dark")
                    except: fig = px.scatter(df, x="Year", y="Value", title="Data Trend", template="plotly_dark")
                    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        except: pass
        return None

    def generate_report(self, topic, region, mode):
        s = load_stats()
        s["prompts_today"] += 1; s["total_prompts"] += 1
        save_stats(s)

        context, resolved_mode, region_code = self._execute_polymorphic_search(topic, region, mode)
        if not context:
            try: context = f"WIKIPEDIA: {wikipedia.summary(topic, sentences=10)}"
            except: return "⚠️ No verifiable data found.", [], None

        instruction = "Structure: <h3>Executive Verdict</h3>, <h3>Deep Dive Analysis</h3>, <h3>Key Evidence</h3>."
        if resolved_mode == "Catalog": instruction = "CATALOG MODE: List EVERY item found. Bullet points."
        elif resolved_mode == "Fact Check": instruction = "VERIFICATION MODE: State VERIFIED or DEBUNKED immediately."

        prompt = f"""
        You are NewsWeave Singularity. TOPIC: {topic} | MODE: {resolved_mode} | REGION: {region}
        DATA: {context}
        INSTRUCTIONS:
        1. {instruction}
        2. **Citations:** <a href='URL' target='_blank' style='color:#00c6ff'>[Source]</a>.
        3. **No Hallucinations:** Verify facts against the vault.
        4. Use HTML tags.
        """
        try: report = self.llm.invoke(prompt).content
        except Exception as e: report = f"AI Error: {e}"

        images = self._smart_image_sweep(topic, region_code)
        chart = self._generate_chart(report)
        return report, images, chart

agent = SingularityAgent()

@app.get("/")
async def serve_interface(request: Request):
    stats = load_stats()
    return templates.TemplateResponse("index.html", {"request": request, "total_likes": stats["total_likes"]})

@app.post("/analyze")
async def analyze_endpoint(request: SearchRequest):
    report, images, chart = agent.generate_report(request.topic, request.region, request.mode)
    return JSONResponse(content={"report": report, "images": images, "chart": chart})

@app.post("/like")
async def like_endpoint():
    s = load_stats()
    s["total_likes"] += 1
    save_stats(s)
    return JSONResponse(content={"new_count": s["total_likes"]})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
