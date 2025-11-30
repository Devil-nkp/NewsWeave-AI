import os
import uvicorn
import re
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

# --- SYSTEM CONFIGURATION ---
# Configure logging to track the agent's "thoughts"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NewsWeave-Brain")

# API Key Strategy: Prioritize Environment Variable, fallback to hardcoded for testing
INTERNAL_API_KEY = os.environ.get("GROQ_API_KEY")

app = FastAPI()

# Mount static folder (for background video) and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- GLOBAL REGION MAPPING (50+ Countries) ---
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

# Input Data Model
class SearchRequest(BaseModel):
    topic: str
    region: str
    mode: str

# ==========================================
# 🧠 OMNI-PRO AGENT v10 (THE CORE)
# ==========================================

class OmniProAgent:
    def __init__(self):
        # Temperature 0.1 ensures high factual accuracy and low hallucination
        self.llm = ChatGroq(temperature=0.1, model_name="llama-3.3-70b-versatile", api_key=INTERNAL_API_KEY)
        self.date_str = datetime.now().strftime("%B %d, %Y")

    def _determine_auto_mode(self, topic):
        """
        Cognitive Router: Analyzes the topic semantics to pick the best mode.
        """
        t = topic.lower()
        if any(x in t for x in ['stock', 'price', 'market', 'growth', 'economy', 'cost', 'revenue', 'gdp']):
            return "Market Analysis"
        if any(x in t for x in ['fake', 'real', 'true', 'hoax', 'scam', 'fact', 'rumor', 'verify']):
            return "Fact Check"
        return "Deep Research"

    def _get_search_strategy(self, topic, mode):
        """
        Strategy Generator: Creates specific search queries based on intent.
        """
        active_mode = self._determine_auto_mode(topic) if mode == "Auto" else mode
        logger.info(f"🧠 Mode Resolved: {active_mode}")

        strategies = []
        if active_mode == "Fact Check":
            strategies = [
                f"is {topic} true or false", 
                f"{topic} official fact check", 
                f"{topic} hoax debunked details"
            ]
        elif active_mode == "Market Analysis":
            strategies = [
                f"{topic} market statistics {datetime.now().year}", 
                f"{topic} financial report analysis", 
                f"{topic} growth trend data"
            ]
        elif active_mode == "Fast":
            strategies = [f"{topic} latest summary news"]
        else: # Deep Research
            strategies = [
                f"{topic} comprehensive analysis", 
                f"{topic} controversy and criticism", 
                f"{topic} timeline of events"
            ]
            
        return strategies, active_mode

    def _smart_image_search(self, topic, region_code):
        """
        Visual Intelligence: Fetches only high-quality, relevant journalistic images.
        Filters out cartoons, vectors, and AI-generated spam.
        """
        search_query = f"{topic} news event photo real"
        gallery = []
        seen_urls = set()
        
        try:
            with DDGS() as ddgs:
                # Fetch 15 to have a buffer for filtering
                results = list(ddgs.images(search_query, region=region_code, max_results=15))
                
                for r in results:
                    title = r.get('title', '').lower()
                    src = r.get('image', '')
                    
                    # --- QUALITY FILTERS ---
                    if src in seen_urls: continue
                    if any(x in title for x in ["cartoon", "vector", "clipart", "icon", "logo", "ai generated"]): 
                        continue
                    
                    gallery.append({"src": src, "title": r['title']})
                    seen_urls.add(src)
                    
                    if len(gallery) >= 4: break # We only need 4 high-quality images
        except Exception as e:
            logger.error(f"Image Search Error: {e}")
            
        return gallery

    def _execute_mission_search(self, topic, region, mode):
        """
        Polymorphic Search Engine: Tries multiple backends to avoid blocking.
        """
        vault = ""
        region_code = REGION_MAP.get(region, "wt-wt")
        strategies, resolved_mode = self._get_search_strategy(topic, mode)
        
        print(f"🕵️ Executing Search Strategies in {region} ({region_code})...")

        try:
            with DDGS() as ddgs:
                for query in strategies:
                    results = []
                    # 1. Try News Backend
                    try:
                        results = list(ddgs.news(query, region=region_code, max_results=4))
                    except:
                        # 2. Fallback to Text Backend (Lite)
                        try:
                            results = list(ddgs.text(query, region=region_code, backend="lite", max_results=4))
                        except: pass

                    # Accumulate Evidence
                    for r in results:
                        title = r.get('title', 'Source')
                        link = r.get('url', r.get('href', '#'))
                        body = r.get('body', r.get('snippet', ''))
                        date = r.get('date', 'Recent')
                        
                        vault += f"SOURCE: {title} ({date})\nLINK: {link}\nEVIDENCE: {body}\n\n"
        except Exception as e:
            logger.error(f"Critical Search Failure: {e}")

        return vault, resolved_mode, region_code

    def generate_report(self, topic, region, mode):
        # 1. GATHER INTELLIGENCE
        context, resolved_mode, region_code = self._execute_mission_search(topic, region, mode)
        
        # 2. FALLBACK PROTOCOL (WIKIPEDIA)
        if not context or len(context) < 100:
            try:
                context = f"WIKIPEDIA SUMMARY: {wikipedia.summary(topic, sentences=6)}"
            except:
                return "⚠️ Mission Failed: No verifiable data found on this topic. Try a more specific query.", []

        # 3. SYNTHESIZE INTELLIGENCE (THE BRAIN)
        # Dynamic prompt injection based on mode
        mode_instruction = "Your goal is COMPREHENSIVE INTELLIGENCE."
        if resolved_mode == "Fact Check":
            mode_instruction = "Your goal is VERIFICATION. Explicitly confirm or debunk the topic based on evidence."
        elif resolved_mode == "Market Analysis":
            mode_instruction = "Your goal is DATA. Focus on financial numbers, stock trends, and economic impact."
        
        prompt = f"""
        You are NewsWeave Omni-Pro (v10).
        
        MISSION PARAMETERS:
        - Topic: {topic}
        - Current Date: {self.date_str}
        - Region Focus: {region}
        - Active Mode: {resolved_mode}
        
        INTELLIGENCE VAULT (RAW DATA):
        {context}
        
        MANDATORY DIRECTIVES:
        1. {mode_instruction}
        2. **Citation Rule:** Every single claim must be backed by a source from the Vault. Format: <a href='URL' target='_blank' style='color:#007bff; text-decoration:none;'>[Source Name]</a>.
        3. **Formatting:** Use HTML tags strictly (<h3> for headers, <p> for text, <ul>/<li> for lists, <strong> for emphasis).
        4. **Tone:** Professional, Forensic, Objective.
        
        REQUIRED REPORT STRUCTURE:
        - <h3>Executive Verdict</h3> (A concise summary of the reality)
        - <h3>Deep Dive Analysis</h3> (Detailed synthesis of the gathered intelligence)
        - <h3>Key Evidence</h3> (Bullet points containing specific numbers, dates, or quotes)
        - <h3>Strategic Outlook</h3> (What happens next? Future implications)
        """
        
        try:
            report_content = self.llm.invoke(prompt).content
        except Exception as e:
            report_content = f"<p style='color:red'>AI Processing Error: {str(e)}</p>"
        
        # 4. GATHER VISUALS
        images = self._smart_image_search(topic, region_code)
        
        return report_content, images

# Initialize Agent
agent = OmniProAgent()

# ==========================================
# 🌐 API ROUTES
# ==========================================

@app.get("/")
async def serve_interface(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_endpoint(request: SearchRequest):
    report, images = agent.generate_report(request.topic, request.region, request.mode)
    return JSONResponse(content={
        "topic": request.topic,
        "report": report,
        "images": images
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
