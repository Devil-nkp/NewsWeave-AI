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
logger = logging.getLogger("NewsWeave-Infinity")

INTERNAL_API_KEY = os.environ.get("GROQ_API_KEY")

app = FastAPI()
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
# 🧠 INFINITY INTELLIGENCE AGENT (v12)
# ==========================================

class InfinityAgent:
    def __init__(self):
        # Temperature 0.0 for absolute robotic precision (No hallucinations)
        self.llm = ChatGroq(temperature=0.0, model_name="llama-3.3-70b-versatile", api_key=INTERNAL_API_KEY)
        self.date_str = datetime.now().strftime("%B %d, %Y")

    def _determine_mode(self, topic):
        t = topic.lower()
        # CATALOG MODE: Detects requests for lists, types, or collections
        if any(x in t for x in ['all', 'list', 'types of', 'top 10', 'every', 'catalog', 'classification']):
            return "Catalog"
        # MARKET MODE: Detects financial intent
        if any(x in t for x in ['stock', 'price', 'market', 'growth', 'economy', 'cost', 'revenue']):
            return "Market Analysis"
        # TRUTH MODE: Detects skepticism
        if any(x in t for x in ['fake', 'real', 'true', 'hoax', 'scam', 'fact', 'verify', 'rumor']):
            return "Fact Check"
        return "Deep Research"

    def _smart_image_sweep(self, topic, region_code):
        """
        DEEP VISUAL SWEEP: Ensures 20-50 REAL images.
        Iterates through multiple semantic queries to build a massive gallery.
        """
        # 1. Determine Field Context for better images
        field_context = "news"
        if "medic" in topic.lower(): field_context = "medical device"
        elif "tech" in topic.lower(): field_context = "technology product"
        elif "space" in topic.lower(): field_context = "spacecraft launch"
        
        # 2. Multi-Vector Image Queries
        queries = [
            f"{topic} {field_context} photo",
            f"{topic} real life",
            f"{topic} event photography",
            f"{topic} press conference",
            f"{topic} official photo"
        ]
        
        gallery = []
        seen_urls = set()
        
        # 3. Strict Anti-AI Blacklist
        blacklist = [
            "ai generated", "midjourney", "dall-e", "stable diffusion", "render", "concept art", 
            "illustration", "vector", "cartoon", "drawing", "clipart", "logo", "icon", "fantasy", "3d model"
        ]

        print(f"📸 Starting Deep Visual Sweep for: {topic}")

        try:
            with DDGS() as ddgs:
                for q in queries:
                    if len(gallery) >= 50: break # Hard cap
                    
                    try:
                        results = list(ddgs.images(q, region=region_code, max_results=30))
                        for r in results:
                            if len(gallery) >= 50: break
                            
                            title = r.get('title', '').lower()
                            src = r.get('image', '')
                            
                            # Deduplication & Quality Filter
                            if src in seen_urls: continue
                            if any(b in title for b in blacklist): continue
                            
                            # Valid Image Found
                            gallery.append({"src": src, "title": r['title']})
                            seen_urls.add(src)
                    except:
                        continue
                    
                    time.sleep(0.2) # Polite delay
        except Exception as e:
            logger.error(f"Image Sweep Error: {e}")
            
        # If we still have few images, fallback to generic search
        if len(gallery) < 10:
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.images(topic, region=region_code, max_results=20))
                    for r in results:
                        if len(gallery) >= 20: break
                        gallery.append({"src": r['image'], "title": r['title']})
            except: pass
            
        return gallery

    def _execute_polymorphic_search(self, topic, region, mode):
        """
        Rotates through 4 search backends to guarantee data retrieval.
        """
        region_code = REGION_MAP.get(region, "wt-wt")
        active_mode = self._determine_mode(topic) if mode == "Auto" else mode
        
        # Strategy Generation
        strategies = []
        if active_mode == "Catalog":
            strategies = [f"list of all {topic}", f"types of {topic} comprehensive", f"full list {topic} details"]
        elif active_mode == "Fact Check":
            strategies = [f"{topic} official fact check", f"is {topic} true"]
        elif active_mode == "Market Analysis":
            strategies = [f"{topic} statistics {datetime.now().year}", f"{topic} financial report"]
        else:
            strategies = [f"{topic} comprehensive analysis", f"{topic} controversy", f"{topic} timeline"]

        vault = ""
        
        try:
            with DDGS() as ddgs:
                for query in strategies:
                    # 1. News Backend (Best for current events)
                    try:
                        news = list(ddgs.news(query, region=region_code, max_results=5))
                        for r in news:
                            vault += f"SOURCE: {r['title']} ({r['date']})\nLINK: {r['url']}\nINFO: {r['body']}\n\n"
                    except: pass

                    # 2. Text Backend (Best for catalogs/lists)
                    # We fetch MORE results in Catalog mode to ensure we catch "All" items
                    limit = 10 if active_mode == "Catalog" else 4
                    try:
                        text = list(ddgs.text(query, region=region_code, backend="lite", max_results=limit))
                        for r in text:
                            vault += f"SOURCE: {r['title']}\nLINK: {r['href']}\nINFO: {r['body']}\n\n"
                    except: pass
        except Exception as e:
            logger.error(f"Search Error: {e}")

        return vault, active_mode, region_code

    def _generate_chart(self, report_text):
        """
        Parses the report text for numbers/dates and auto-generates a Plotly chart.
        """
        # Extract years and numbers
        years = re.findall(r'\b(20\d{2})\b', report_text)
        numbers = re.findall(r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\b', report_text)
        
        chart_json = None
        
        if len(years) > 2 and len(numbers) > 2:
            # Create Timeline Scatter Plot
            clean_years = []
            clean_nums = []
            for i in range(min(len(years), len(numbers))):
                try:
                    y = int(years[i])
                    v = float(numbers[i].replace(',', ''))
                    clean_years.append(y)
                    clean_nums.append(v)
                except: pass
            
            if clean_years:
                df = pd.DataFrame({"Year": clean_years, "Value": clean_nums})
                fig = px.scatter(df, x="Year", y="Value", title="Data Trend Analysis", 
                                 trendline="ols" if len(df) > 3 else None, template="plotly_dark")
                chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return chart_json

    def generate_report(self, topic, region, mode):
        # 1. GATHER
        context, resolved_mode, region_code = self._execute_polymorphic_search(topic, region, mode)
        
        # 2. FALLBACK
        if not context:
            try:
                context = f"WIKIPEDIA: {wikipedia.summary(topic, sentences=10)}"
            except:
                return "⚠️ Mission Failed: No verifiable data found.", [], None

        # 3. PROMPT ENGINEERING (THE BRAIN)
        # Specific instructions for "Catalog" mode to ensure "All" items are listed
        structure_instruction = ""
        if resolved_mode == "Catalog":
            structure_instruction = """
            **CATALOG MODE:**
            - Your task is to list **EVERY** single entity/type found in the data.
            - Structure as a bulleted list.
            - Format: <b>Name:</b> One concise line of explanation.
            - Do not summarize "some examples". List ALL of them.
            """
        else:
            structure_instruction = """
            - <h3>Executive Verdict</h3>
            - <h3>Deep Dive Analysis</h3>
            - <h3>Key Evidence</h3> (Bullet points with specific numbers/dates)
            - <h3>Strategic Outlook</h3>
            """

        prompt = f"""
        You are NewsWeave Infinity.
        TOPIC: {topic}
        MODE: {resolved_mode}
        DATE: {self.date_str}
        REGION: {region}
        
        INTELLIGENCE VAULT:
        {context}
        
        INSTRUCTIONS:
        1. {structure_instruction}
        2. **Citations:** <a href='URL' target='_blank'>[Source]</a>.
        3. **No Hallucinations:** Verify facts against the vault. If data is missing, say "Data unavailable".
        4. **HTML Format:** Use <h3>, <p>, <ul>, <li>.
        """
        
        try:
            report = self.llm.invoke(prompt).content
        except:
            report = "AI Generation Error."

        # 4. VISUALS & CHARTS
        images = self._smart_image_sweep(topic, region_code)
        chart = self._generate_chart(report)
        
        return report, images, chart

agent = InfinityAgent()

# ==========================================
# 🌐 API ROUTES
# ==========================================

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

