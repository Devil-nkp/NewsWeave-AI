import os
import uvicorn
import re
import time
import random
import json
import logging
from datetime import datetime
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
logger = logging.getLogger("NewsWeave-Singularity")

# Ensure you set this in Render Environment Variables
INTERNAL_API_KEY = os.environ.get("GROQ_API_KEY")

app = FastAPI()

# Ensure directories exist before mounting
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- GLOBAL REGION MAP (50+ Nations) ---
REGION_MAP = {
    "Global": "wt-wt", "USA": "us-en", "India": "in-en", "UK": "uk-en",
    "Argentina": "ar-es", "Australia": "au-en", "Austria": "at-de",
    "Belgium (fr)": "be-fr", "Belgium (nl)": "be-nl", "Brazil": "br-pt",
    "Bulgaria": "bg-bg", "Canada (en)": "ca-en", "Canada (fr)": "ca-fr",
    "Chile": "cl-es", "China": "cn-zh", "Colombia": "co-es", "Croatia": "hr-hr",
    "Czech Republic": "cz-cs", "Denmark": "dk-da", "Estonia": "ee-et",
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
#  SINGULARITY INTELLIGENCE AGENT (v13)
# ==========================================

class SingularityAgent:
    def __init__(self):
        # Temperature 0.0 for robotic factual precision
        if not INTERNAL_API_KEY:
            logger.warning("GROQ_API_KEY is missing!")
            self.llm = None
        else:
            self.llm = ChatGroq(temperature=0.0, model_name="llama-3.3-70b-versatile", api_key=INTERNAL_API_KEY)
        self.date_str = datetime.now().strftime("%B %d, %Y")

    def _determine_mode(self, topic):
        t = topic.lower()
        if any(x in t for x in ['all', 'list', 'types of', 'top 10', 'top 20', 'top 50', 'every', 'catalog', 'classification', 'examples']):
            return "Catalog"
        if any(x in t for x in ['stock', 'price', 'market', 'growth', 'economy', 'cost', 'revenue', 'finance']):
            return "Market Analysis"
        if any(x in t for x in ['fake', 'real', 'true', 'hoax', 'scam', 'fact', 'verify', 'rumor', 'debunk']):
            return "Fact Check"
        return "Deep Research"

    def _smart_image_sweep(self, topic, region_code):
        context_keyword = "news"
        t_lower = topic.lower()
        if "crime" in t_lower: context_keyword = "police investigation scene"
        elif "tech" in t_lower: context_keyword = "product demonstration"
        elif "medic" in t_lower: context_keyword = "medical device"
        elif "space" in t_lower: context_keyword = "launch pad"
        elif "war" in t_lower: context_keyword = "conflict zone journalism"
        
        queries = [
            f"{topic} {context_keyword} photo",
            f"{topic} real life photography",
            f"{topic} official event",
            f"{topic} press conference",
            f"{topic} close up photo"
        ]
        
        gallery = []
        seen_urls = set()
        
        blacklist = [
            "ai generated", "midjourney", "dall-e", "stable diffusion", "render", "concept art", 
            "illustration", "vector", "cartoon", "drawing", "clipart", "logo", "icon", "fantasy", "3d model", "anime"
        ]

        print(f"📸 Starting Visual Trawl for: {topic}")

        try:
            with DDGS() as ddgs:
                for q in queries:
                    if len(gallery) >= 50: break
                    try:
                        results = list(ddgs.images(q, region=region_code, max_results=30))
                        for r in results:
                            if len(gallery) >= 50: break
                            title = r.get('title', '').lower()
                            src = r.get('image', '')
                            if src in seen_urls: continue
                            if any(b in title for b in blacklist): continue
                            gallery.append({"src": src, "title": r['title']})
                            seen_urls.add(src)
                    except Exception as img_err:
                        logger.warning(f"Image batch failed: {img_err}")
                        continue
                    time.sleep(0.2)
        except Exception as e:
            logger.error(f"Image Sweep Critical Error: {e}")
            
        if len(gallery) < 10:
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.images(topic, region=region_code, max_results=25))
                    for r in results:
                        if len(gallery) >= 25: break
                        if r['image'] not in seen_urls:
                            gallery.append({"src": r['image'], "title": r['title']})
                            seen_urls.add(r['image'])
            except: pass
            
        return gallery

    def _execute_polymorphic_search(self, topic, region, mode):
        region_code = REGION_MAP.get(region, "wt-wt")
        active_mode = self._determine_mode(topic) if mode == "Auto" else mode
        
        strategies = []
        if active_mode == "Catalog":
            strategies = [f"list of all {topic}", f"comprehensive list {topic}", f"types of {topic} with description", f"full classification {topic}"]
        elif active_mode == "Fact Check":
            strategies = [f"{topic} official fact check", f"is {topic} true", f"{topic} hoax debunked"]
        elif active_mode == "Market Analysis":
            strategies = [f"{topic} statistics {datetime.now().year}", f"{topic} market report", f"{topic} revenue data"]
        else:
            strategies = [f"{topic} comprehensive analysis", f"{topic} controversy and criticism", f"{topic} timeline of events", f"{topic} official data"]

        vault = ""
        try:
            with DDGS() as ddgs:
                for query in strategies:
                    try:
                        news = list(ddgs.news(query, region=region_code, max_results=5))
                        for r in news:
                            vault += f"SOURCE: {r['title']} ({r['date']})\nLINK: {r['url']}\nINFO: {r['body']}\n\n"
                    except: pass
                    limit = 15 if active_mode == "Catalog" else 5
                    try:
                        text = list(ddgs.text(query, region=region_code, backend="lite", max_results=limit))
                        for r in text:
                            vault += f"SOURCE: {r['title']}\nLINK: {r['href']}\nINFO: {r['body']}\n\n"
                    except: pass
        except Exception as e:
            logger.error(f"Search Error: {e}")

        return vault, active_mode, region_code

    def _generate_chart(self, report_text):
        try:
            years = re.findall(r'\b(20\d{2})\b', report_text)
            numbers = re.findall(r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\b', report_text)
            
            if len(years) > 2 and len(numbers) > 2:
                clean_years, clean_nums = [], []
                for i in range(min(len(years), len(numbers))):
                    try:
                        y = int(years[i])
                        v = float(numbers[i].replace(',', ''))
                        clean_years.append(y)
                        clean_nums.append(v)
                    except: pass
                
                if clean_years:
                    df = pd.DataFrame({"Year": clean_years, "Value": clean_nums})
                    # Trendline removed to avoid 'statsmodels' dependency crash on small servers
                    fig = px.scatter(df, x="Year", y="Value", title="Data Trend Analysis", template="plotly_dark")
                    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        except Exception as e:
            logger.error(f"Chart Error: {e}")
        return None

    def generate_report(self, topic, region, mode):
        if not self.llm:
            return "<h3>System Error</h3><p>API Key missing. Please configure GROQ_API_KEY in Render settings.</p>", [], None

        context, resolved_mode, region_code = self._execute_polymorphic_search(topic, region, mode)
        
        if not context:
            try:
                context = f"WIKIPEDIA: {wikipedia.summary(topic, sentences=10)}"
            except:
                return " Mission Failed: No verifiable data found.", [], None

        structure_instruction = ""
        if resolved_mode == "Catalog":
            structure_instruction = "CATALOG MODE: Generate an EXHAUSTIVE LIST using bullet points."
        elif resolved_mode == "Fact Check":
            structure_instruction = "FACT CHECK: Truth Verdict, Reality Check, Evidence Audit."
        else:
            structure_instruction = "DEEP RESEARCH: Executive Verdict, Deep Dive, Key Evidence, Strategic Outlook."

        prompt = f"""
        You are NewsWeave Singularity. 
        TOPIC: {topic} | MODE: {resolved_mode}
        DATE: {self.date_str} | REGION: {region}
        
        INTELLIGENCE VAULT:
        {context}
        
        INSTRUCTIONS:
        1. {structure_instruction}
        2. Format using HTML tags (<h3>, <b>, <ul>, <li>).
        3. No Hallucinations.
        """
        
        try:
            report = self.llm.invoke(prompt).content
        except Exception as e:
            report = f"<p style='color:red'>Error generating report: {str(e)}</p>"

        images = self._smart_image_sweep(topic, region_code)
        chart = self._generate_chart(report)
        return report, images, chart

agent = SingularityAgent()

@app.get("/")
async def serve_interface(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_endpoint(request: SearchRequest):
    report, images, chart = agent.generate_report(request.topic, request.region, request.mode)
    return JSONResponse(content={
        "topic": request.topic,
        "report": report,
        "images": images,
        "chart": chart
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
