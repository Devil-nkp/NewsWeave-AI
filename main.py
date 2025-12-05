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

# Region Map
REGION_MAP = {
    "Global": "wt-wt", "USA": "us-en", "India": "in-en", "UK": "uk-en",
    "China": "cn-zh", "Japan": "jp-jp", "Germany": "de-de", "France": "fr-fr",
    "Russia": "ru-ru", "Brazil": "br-pt", "Canada": "ca-en"
}

class SearchRequest(BaseModel):
    topic: str
    region: str
    mode: str

class InfinityAgent:
    def __init__(self):
        self.llm = ChatGroq(temperature=0.0, model_name="llama-3.3-70b-versatile", api_key=INTERNAL_API_KEY)
        self.date_str = datetime.now().strftime("%B %d, %Y")

    def _determine_mode(self, topic):
        t = topic.lower()
        if any(x in t for x in ['all', 'list', 'types of', 'top 10', 'top 20', 'every', 'catalog']):
            return "Catalog"
        if any(x in t for x in ['stock', 'price', 'market', 'growth', 'economy', 'revenue']):
            return "Market Analysis"
        if any(x in t for x in ['fake', 'real', 'true', 'hoax', 'fact', 'verify']):
            return "Fact Check"
        return "Deep Research"

    def _smart_image_sweep(self, topic, region_code):
        # Error handling wrapper for images
        try:
            queries = [f"{topic} news photo", f"{topic} event", f"{topic} official"]
            gallery = []
            seen = set()
            blacklist = ["ai generated", "cartoon", "vector", "drawing", "clipart", "logo"]

            with DDGS() as ddgs:
                for q in queries:
                    if len(gallery) >= 20: break
                    try:
                        # Reduced max_results per query to prevent timeouts
                        results = list(ddgs.images(q, region=region_code, max_results=15))
                        for r in results:
                            if len(gallery) >= 20: break
                            t = r.get('title', '').lower()
                            src = r.get('image', '')
                            if src not in seen and not any(b in t for b in blacklist):
                                gallery.append({"src": src, "title": r['title']})
                                seen.add(src)
                    except: continue
            return gallery
        except Exception as e:
            logger.error(f"Image Sweep Error: {e}")
            return []

    def _execute_polymorphic_search(self, topic, region, mode):
        region_code = REGION_MAP.get(region, "wt-wt")
        active_mode = self._determine_mode(topic) if mode == "Auto" else mode
        vault = ""
        
        strategies = [f"{topic} news", f"{topic} analysis", f"{topic} statistics"]
        if active_mode == "Catalog":
            strategies = [f"list of {topic}", f"top {topic}", f"{topic} examples"]

        try:
            with DDGS() as ddgs:
                for query in strategies:
                    try:
                        # Prioritize News backend
                        results = list(ddgs.news(query, region=region_code, max_results=5))
                        for r in results:
                            vault += f"SOURCE: {r['title']} ({r['date']})\nLINK: {r['url']}\nINFO: {r['body']}\n\n"
                    except: 
                        # Fallback to Text backend
                        try:
                            results = list(ddgs.text(query, region=region_code, max_results=4))
                            for r in results:
                                vault += f"SOURCE: {r['title']}\nLINK: {r['href']}\nINFO: {r['body']}\n\n"
                        except: pass
        except Exception as e:
            logger.error(f"Search Error: {e}")

        return vault, active_mode, region_code

    def _generate_chart(self, report_text):
        """
        Robust Chart Generator - Won't crash the server if math fails.
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
                    # Removed 'trendline="ols"' to remove dependency on statsmodels and increase stability
                    fig = px.scatter(df, x="Year", y="Value", title="Trend Analysis", template="plotly_dark")
                    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        except Exception as e:
            logger.error(f"Chart Error: {e}")
        return None

    def generate_report(self, topic, region, mode):
        context, resolved_mode, region_code = self._execute_polymorphic_search(topic, region, mode)
        
        if not context:
            try:
                context = f"WIKIPEDIA: {wikipedia.summary(topic, sentences=6)}"
            except:
                return "⚠️ Mission Failed: No verifiable data found.", [], None

        structure_instruction = """
        - <h3>Executive Verdict</h3>
        - <h3>Deep Dive Analysis</h3>
        - <h3>Key Evidence</h3> (Bullet points with specific numbers/dates)
        - <h3>Strategic Outlook</h3>
        """
        if resolved_mode == "Catalog":
            structure_instruction = """
            **CATALOG MODE:**
            - List EVERY single entity/type found.
            - Format: <b>Name:</b> One concise line of explanation.
            """

        prompt = f"""
        You are NewsWeave Infinity. TOPIC: {topic} | MODE: {resolved_mode}
        DATE: {self.date_str} | REGION: {region}
        
        INTELLIGENCE VAULT:
        {context}
        
        INSTRUCTIONS:
        1. {structure_instruction}
        2. **Citations:** <a href='URL' target='_blank'>[Source]</a>.
        3. **No Hallucinations:** Verify facts against the vault.
        4. **HTML Format:** Use <h3>, <p>, <ul>, <li>.
        """
        
        try:
            report = self.llm.invoke(prompt).content
        except Exception as e:
            report = f"<p style='color:red'>Error generating report: {str(e)}</p>"

        images = self._smart_image_sweep(topic, region_code)
        chart = self._generate_chart(report)
        
        return report, images, chart

agent = InfinityAgent()

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

