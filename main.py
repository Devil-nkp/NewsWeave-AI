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
#  SINGULARITY INTELLIGENCE AGENT (v13)
# ==========================================

class SingularityAgent:
    def __init__(self):
        # Temperature 0.0 for robotic factual precision
        self.llm = ChatGroq(temperature=0.0, model_name="llama-3.3-70b-versatile", api_key=INTERNAL_API_KEY)
        self.date_str = datetime.now().strftime("%B %d, %Y")

    def _determine_mode(self, topic):
        t = topic.lower()
        # CATALOG MODE: Detects requests for lists, types, or collections
        if any(x in t for x in ['all', 'list', 'types of', 'top 10', 'top 20', 'top 50', 'every', 'catalog', 'classification', 'examples']):
            return "Catalog"
        # MARKET MODE: Detects financial intent
        if any(x in t for x in ['stock', 'price', 'market', 'growth', 'economy', 'cost', 'revenue', 'finance']):
            return "Market Analysis"
        # TRUTH MODE: Detects skepticism
        if any(x in t for x in ['fake', 'real', 'true', 'hoax', 'scam', 'fact', 'verify', 'rumor', 'debunk']):
            return "Fact Check"
        return "Deep Research"

    def _smart_image_sweep(self, topic, region_code):
        """
        VISUAL TRAWL ENGINE: Aggressively hunts for 20-50 REAL images.
        Uses domain-specific context injection.
        """
        # 1. Determine Context for better search
        context_keyword = "news"
        t_lower = topic.lower()
        if "crime" in t_lower: context_keyword = "police investigation scene"
        elif "tech" in t_lower: context_keyword = "product demonstration"
        elif "medic" in t_lower: context_keyword = "medical device"
        elif "space" in t_lower: context_keyword = "launch pad"
        elif "war" in t_lower: context_keyword = "conflict zone journalism"
        
        # 2. Multi-Vector Visual Queries
        queries = [
            f"{topic} {context_keyword} photo",
            f"{topic} real life photography",
            f"{topic} official event",
            f"{topic} press conference",
            f"{topic} close up photo"
        ]
        
        gallery = []
        seen_urls = set()
        
        # 3. Strict Anti-AI / Anti-Cartoon Firewall
        blacklist = [
            "ai generated", "midjourney", "dall-e", "stable diffusion", "render", "concept art", 
            "illustration", "vector", "cartoon", "drawing", "clipart", "logo", "icon", "fantasy", "3d model", "anime"
        ]

        print(f"📸 Starting Visual Trawl for: {topic}")

        try:
            with DDGS() as ddgs:
                for q in queries:
                    if len(gallery) >= 50: break # Cap at 50
                    
                    try:
                        # Fetch large batch
                        results = list(ddgs.images(q, region=region_code, max_results=30))
                        for r in results:
                            if len(gallery) >= 50: break
                            
                            title = r.get('title', '').lower()
                            src = r.get('image', '')
                            
                            # Deduplication
                            if src in seen_urls: continue
                            # Content Filter
                            if any(b in title for b in blacklist): continue
                            
                            gallery.append({"src": src, "title": r['title']})
                            seen_urls.add(src)
                    except Exception as img_err:
                        logger.warning(f"Image batch failed: {img_err}")
                        continue
                    
                    time.sleep(0.2) # Anti-rate-limit pause
        except Exception as e:
            logger.error(f"Image Sweep Critical Error: {e}")
            
        # Fallback: If < 10 images, try a very broad search
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
        """
        Rotates through 4 search backends + Conflict Resolution Vector.
        """
        region_code = REGION_MAP.get(region, "wt-wt")
        active_mode = self._determine_mode(topic) if mode == "Auto" else mode
        
        strategies = []
        if active_mode == "Catalog":
            strategies = [
                f"list of all {topic}", 
                f"comprehensive list {topic}", 
                f"types of {topic} with description",
                f"full classification {topic}"
            ]
        elif active_mode == "Fact Check":
            strategies = [f"{topic} official fact check", f"is {topic} true", f"{topic} hoax debunked"]
        elif active_mode == "Market Analysis":
            strategies = [f"{topic} statistics {datetime.now().year}", f"{topic} market report", f"{topic} revenue data"]
        else:
            # Deep Research: Includes Conflict Vector
            strategies = [
                f"{topic} comprehensive analysis", 
                f"{topic} controversy and criticism", # Finds conflicts
                f"{topic} timeline of events",
                f"{topic} official data"
            ]

        vault = ""
        
        try:
            with DDGS() as ddgs:
                for query in strategies:
                    # 1. News Backend
                    try:
                        news = list(ddgs.news(query, region=region_code, max_results=5))
                        for r in news:
                            vault += f"SOURCE: {r['title']} ({r['date']})\nLINK: {r['url']}\nINFO: {r['body']}\n\n"
                    except: pass

                    # 2. Text Backend (High Volume for Catalogs)
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
        """
        Robust Chart Generator with statsmodels support.
        """
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
                    # Trendline enabled (requires statsmodels)
                    fig = px.scatter(df, x="Year", y="Value", title="Data Trend Analysis", 
                                     trendline="ols" if len(df) > 3 else None, template="plotly_dark")
                    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        except Exception as e:
            logger.error(f"Chart Error: {e}")
        return None

    def generate_report(self, topic, region, mode):
        # 1. GATHER
        context, resolved_mode, region_code = self._execute_polymorphic_search(topic, region, mode)
        
        # 2. FALLBACK
        if not context:
            try:
                context = f"WIKIPEDIA: {wikipedia.summary(topic, sentences=10)}"
            except:
                return " Mission Failed: No verifiable data found.", [], None

        # 3. PROMPT ENGINEERING (THE BRAIN)
        structure_instruction = ""
        
        if resolved_mode == "Catalog":
            structure_instruction = """
            **CATALOG MODE ACTIVATED:**
            - You MUST generate an **EXHAUSTIVE LIST** of items.
            - Do not group them into paragraphs. Use Bullet Points.
            - **Format:** <b>Item Name:</b> One concise sentence explaining it.
            - If the user asked for "All", list as many as found in the data (up to 50).
            - Do not omit items.
            """
        elif resolved_mode == "Fact Check":
             structure_instruction = """
             - <h3>Truth Verdict</h3> (Verified/Debunked)
             - <h3>Reality Check</h3> (What actually happened)
             - <h3>Evidence Audit</h3>
             """
        else:
            structure_instruction = """
            - <h3>Executive Verdict</h3>
            - <h3>Deep Dive Analysis</h3> (Include Conflict Analysis: Side A vs Side B)
            - <h3>Key Evidence</h3> (Bullet points with numbers)
            - <h3>Strategic Outlook</h3>
            """

        prompt = f"""
        You are NewsWeave Singularity. 
        TOPIC: {topic} | MODE: {resolved_mode}
        DATE: {self.date_str} | REGION: {region}
        
        INTELLIGENCE VAULT:
        {context}
        
        INSTRUCTIONS:
        1. {structure_instruction}
        2. **Citations:** <a href='URL' target='_blank' style='color:#00c6ff'>[Source]</a>.
        3. **No Hallucinations:** Verify facts against the vault. If data is missing, say "Data unavailable".
        4. **Conflict Handling:** If sources disagree, state "Conflict Detected: Source A says X, while Source B says Y".
        """
        
        try:
            report = self.llm.invoke(prompt).content
        except Exception as e:
            report = f"<p style='color:red'>Error generating report: {str(e)}</p>"

        # 4. VISUALS & CHARTS
        images = self._smart_image_sweep(topic, region_code)
        chart = self._generate_chart(report)
        
        return report, images, chart

agent = SingularityAgent()

# ==========================================
#  API ROUTES
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
