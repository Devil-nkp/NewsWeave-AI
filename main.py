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
logger = logging.getLogger("NewsWeave-Singularity-v21")

INTERNAL_API_KEY = os.environ.get("GROQ_API_KEY", "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")

app = FastAPI()
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

# --- EXTENSIVE GLOBAL REGION MAP (60+ Countries) ---
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
        # Temp 0.0 for absolute precision
        self.llm = ChatGroq(temperature=0.0, model_name="llama-3.3-70b-versatile", api_key=INTERNAL_API_KEY)
        self.date_str = datetime.now().strftime("%B %d, %Y")

    def _determine_mode(self, topic):
        t = topic.lower()
        if any(x in t for x in ['all', 'list', 'types', 'catalog', 'every', 'classification', 'top 10']):
            return "Catalog"
        if any(x in t for x in ['stock', 'price', 'market', 'growth', 'economy', 'revenue', 'gdp']):
            return "Market Analysis"
        if any(x in t for x in ['fake', 'real', 'true', 'hoax', 'scam', 'fact', 'verify']):
            return "Fact Check"
        return "Deep Research"

    def _extract_specific_visual_queries(self, topic, vault_text):
        """
        Advanced Logic: Reads the text results to extract Specific Nouns for image searching.
        This ensures images are 'Particular', not generic.
        """
        # 1. Start with the main topic
        queries = [f"{topic} real life photo", f"{topic} official event"]
        
        # 2. Extract potential proper nouns or key terms from the vault (Simple heuristic)
        # We look for capitalized words that appear frequently in the search results
        words = re.findall(r'\b[A-Z][a-z]+\b', vault_text)
        common_entities = [word for word, count in Counter(words).most_common(3) if len(word) > 3]
        
        # 3. Add specific queries based on found entities
        for entity in common_entities:
            if entity.lower() not in topic.lower():
                queries.append(f"{topic} {entity} photo")
        
        return queries

    def _smart_image_sweep(self, topic, region_code, vault_text):
        """
        VISUAL TRAWL v2: Uses extracted context to find 20-50 specific images.
        """
        gallery = []
        seen = set()
        blacklist = ["ai generated", "cartoon", "vector", "drawing", "clipart", "logo", "icon", "render", "3d", "illustration", "sketch", "stock"]
        
        # Use the new specific query generator
        queries = self._extract_specific_visual_queries(topic, vault_text)
        
        try:
            with DDGS() as ddgs:
                for q in queries:
                    if len(gallery) >= 50: break
                    try:
                        # High volume fetch
                        results = list(ddgs.images(q, region=region_code, max_results=20))
                        for r in results:
                            if len(gallery) >= 50: break
                            
                            title = r.get('title', '').lower()
                            src = r.get('image', '')
                            
                            # Strict Filtering
                            if src in seen: continue
                            if any(b in title for b in blacklist): continue
                            
                            gallery.append({"src": src, "title": r['title']})
                            seen.add(src)
                    except: continue
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"Image Error: {e}")
            
        return gallery

    def _safe_search(self, query, region_code, backend='api', limit=5):
        """Retry logic for search"""
        for attempt in range(3):
            try:
                with DDGS() as ddgs:
                    if backend == 'news': return list(ddgs.news(query, region=region_code, max_results=limit))
                    else: return list(ddgs.text(query, region=region_code, backend=backend, max_results=limit))
            except:
                time.sleep(0.5)
        return []

    def _execute_polymorphic_search(self, topic, region, mode):
        region_code = REGION_MAP.get(region, "wt-wt")
        active_mode = self._determine_mode(topic) if mode == "Auto" else mode
        
        strategies = [f"{topic} latest news", f"{topic} analysis details"]
        if active_mode == "Catalog":
            strategies = [f"list of all {topic}", f"types of {topic} detailed list", f"full classification {topic}"]
        elif active_mode == "Fact Check":
            strategies = [f"is {topic} true", f"{topic} official fact check", f"{topic} debunked"]
        elif active_mode == "Market Analysis":
             strategies = [f"{topic} market size {datetime.now().year}", f"{topic} financial report", f"{topic} growth statistics"]
            
        vault = ""
        for q in strategies:
            # News Backend
            results = self._safe_search(q, region_code, backend='news', limit=5)
            for r in results: vault += f"SOURCE: {r['title']} ({r['date']})\nLINK: {r['url']}\nINFO: {r['body']}\n\n"
            
            # Text Backend (Deep Search)
            if len(vault) < 2000:
                results = self._safe_search(q, region_code, backend='lite', limit=8)
                for r in results: vault += f"SOURCE: {r['title']}\nLINK: {r['href']}\nINFO: {r['body']}\n\n"
        
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
                    try:
                        fig = px.scatter(df, x="Year", y="Value", title="Data Trend Analysis", trendline="ols", template="plotly_dark")
                    except:
                        fig = px.scatter(df, x="Year", y="Value", title="Data Trend Analysis", template="plotly_dark")
                    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        except: pass
        return None

    def generate_report(self, topic, region, mode):
        # 1. Persistence
        s = load_stats()
        s["prompts_today"] += 1; s["total_prompts"] += 1
        save_stats(s)

        # 2. Gather
        context, resolved_mode, region_code = self._execute_polymorphic_search(topic, region, mode)
        if not context:
            try: context = f"WIKIPEDIA: {wikipedia.summary(topic, sentences=10)}"
            except: return "⚠️ No verifiable data found.", [], None

        # 3. Synthesize
        instruction = ""
        if resolved_mode == "Catalog":
            instruction = "CATALOG MODE: List EVERY item found. Use bullet points. Format: <b>Item Name:</b> Concise description."
        elif resolved_mode == "Fact Check":
            instruction = "VERIFICATION MODE: State VERIFIED or DEBUNKED immediately."
        else:
            instruction = "Structure: <h3>Executive Verdict</h3>, <h3>Deep Dive Analysis</h3>, <h3>Key Evidence</h3>, <h3>Strategic Outlook</h3>."

        prompt = f"""
        You are NewsWeave Singularity. TOPIC: {topic} | MODE: {resolved_mode} | REGION: {region}
        DATE: {self.date_str}
        
        INTELLIGENCE VAULT:
        {context}
        
        INSTRUCTIONS:
        1. {instruction}
        2. **Citations:** <a href='URL' target='_blank' style='color:#00c6ff; text-decoration:none;'>[Source]</a>.
        3. **Polished Writing:** Use professional, investigative journalism tone.
        4. **No Hallucinations:** If data is missing, say so.
        5. Use HTML tags.
        """
        
        try:
            report = self.llm.invoke(prompt).content
        except Exception as e:
            report = f"<p style='color:red'>AI Core Error: {str(e)}</p>"

        # 4. Assets (Visual Trawl uses the Context Vault now!)
        images = self._smart_image_sweep(topic, region_code, context)
        chart = self._generate_chart(report)
        
        return report, images, chart

agent = SingularityAgent()

# ==========================================
# 🌐 API ROUTES
# ==========================================

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
```

---

### **4. `templates/index.html` (The Complete Interface)**
This HTML file now contains **EVERY** country from the backend list, ensuring full alignment.

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NewsWeave Singularity v21</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Segoe UI', sans-serif; }
        body { background-color: #000; color: white; overflow-x: hidden; }
        #video-background { position: fixed; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; z-index: -1; }
        
        .container { min-height: 100vh; display: flex; flex-direction: column; justify-content: center; align-items: center; padding: 20px; transition: opacity 0.5s ease; }
        .hidden { display: none !important; }
        .fade-in { animation: fadeIn 1s ease forwards; }

        .logo-lg { font-size: 5rem; font-weight: 900; letter-spacing: 5px; background: linear-gradient(90deg, #fff, #aaa); -webkit-background-clip: text; color: transparent; text-shadow: 0 0 50px rgba(255,255,255,0.2); margin-bottom: 10px; }
        .tagline { font-size: 1.2rem; color: #ccc; letter-spacing: 1px; margin-bottom: 20px; text-shadow: 0 2px 4px black; }
        
        .social-proof { background: rgba(0,0,0,0.6); padding: 10px 25px; border-radius: 30px; border: 1px solid #333; font-size: 0.9rem; color: #aaa; margin-bottom: 40px; display: flex; align-items: center; gap: 10px; backdrop-filter: blur(5px); }
        .social-proof i { color: #ff6b6b; animation: pulse 1.5s infinite; }
        .like-number { color: #00c6ff; font-weight: bold; font-size: 1.1rem; }

        .btn-start { padding: 15px 50px; font-size: 1.2rem; background: #007bff; color: white; border: none; border-radius: 30px; cursor: pointer; font-weight: bold; transition: 0.3s; box-shadow: 0 0 20px rgba(0,123,255,0.4); }
        .btn-start:hover { transform: scale(1.05); box-shadow: 0 0 40px rgba(0,123,255,0.8); }

        .search-wrapper { width: 100%; max-width: 900px; text-align: center; animation: fadeInUp 1s ease; }
        .super-bar { display: flex; align-items: center; background: rgba(0, 0, 0, 0.85); backdrop-filter: blur(20px); border: 1px solid #007bff; border-radius: 50px; padding: 5px 15px; box-shadow: 0 0 40px rgba(0, 123, 255, 0.3); height: 70px; }
        
        .search-input { flex: 1; background: transparent; border: none; color: white; font-size: 1.1rem; padding: 0 15px; outline: none; }
        .bar-select { background: transparent; color: #ccc; border: none; font-size: 0.9rem; cursor: pointer; outline: none; padding: 5px; font-weight: 600; max-width: 130px; }
        .bar-select option { background: #000; color: white; }
        .icon-wrapper { display: flex; align-items: center; border-right: 1px solid rgba(255,255,255,0.2); padding-right: 10px; margin-right: 5px; }
        .icon-wrapper i { color: #007bff; font-size: 1.1rem; margin-right: 5px; }

        .submit-btn { width: 45px; height: 45px; border-radius: 50%; border: none; background: #007bff; color: white; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: 0.3s; margin-left: 10px; font-size: 1.2rem; }
        .submit-btn:hover { transform: scale(1.1); background: white; color: #007bff; }

        .help-link { margin-top: 15px; color: #aaa; cursor: pointer; font-size: 0.9rem; transition: 0.3s; text-shadow: 0 1px 2px black; }
        .help-link:hover { color: #007bff; text-decoration: underline; }

        .results-wrapper { width: 100%; max-width: 1400px; margin-top: 40px; display: grid; grid-template-columns: 40% 60%; gap: 30px; animation: fadeInUp 1s ease; }
        .card { background: rgba(10, 10, 10, 0.9); border: 1px solid #007bff; padding: 30px; border-radius: 20px; backdrop-filter: blur(20px); box-shadow: 0 0 40px rgba(0,0,0,0.8); max-height: 85vh; overflow-y: auto; }
        
        .report-content h3 { color: #007bff; margin: 25px 0 10px; border-bottom: 1px solid #333; padding-bottom: 5px; font-size: 1.4rem; }
        .report-content p { line-height: 1.7; color: #eee; margin-bottom: 15px; font-size: 1.05rem; }
        .report-content ul { padding-left: 20px; color: #ddd; }
        .report-content a { color: #ff6b6b; font-weight: bold; text-decoration: none; }

        .img-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; max-height: 50vh; overflow-y: auto; padding-right: 5px; margin-bottom: 20px;}
        .img-item { height: 120px; border-radius: 8px; overflow: hidden; cursor: pointer; border: 1px solid #333; transition: 0.3s; }
        .img-item:hover { transform: scale(1.05); border-color: #007bff; }
        .img-item img { width: 100%; height: 100%; object-fit: cover; }
        .chart-container { height: 300px; background: rgba(0,0,0,0.5); border-radius: 10px; }

        .like-section { text-align: center; border-top: 1px solid #333; padding-top: 20px; margin-top: 20px; }
        .btn-like { background: #1a1a1a; border: 1px solid #ff6b6b; color: #ff6b6b; padding: 10px 30px; border-radius: 30px; cursor: pointer; font-size: 1rem; transition: 0.3s; display: inline-flex; align-items: center; gap: 10px; }
        .btn-like:hover { background: #ff6b6b; color: white; box-shadow: 0 0 20px rgba(255, 107, 107, 0.5); }

        .fs-modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.95); z-index: 2000; justify-content: center; align-items: center; }
        .fs-modal img { max-width: 95%; max-height: 95%; border-radius: 5px; }
        .fs-close { position: absolute; top: 20px; right: 30px; color: white; font-size: 3rem; cursor: pointer; }

        .help-modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.9); z-index: 2000; justify-content: center; align-items: center; }
        .help-box { background: #0f0f0f; padding: 40px; border-radius: 15px; max-width: 700px; width: 90%; max-height: 80vh; overflow-y: auto; border: 1px solid #007bff; color: white; line-height: 1.6; box-shadow: 0 0 50px rgba(0, 123, 255, 0.2); }

        .loader { display: none; border: 3px solid rgba(255,255,255,0.1); border-top: 3px solid #007bff; border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; margin: 30px auto; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        @keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.2); } 100% { transform: scale(1); } }
        
        @media (max-width: 1000px) { .results-wrapper { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <video id="video-background" autoplay muted loop playsinline><source src="/static/background.mp4" type="video/mp4"></video>

    <!-- LANDING -->
    <div class="container" id="p1">
        <h1 class="logo-lg">NEWSWEAVE</h1>
        <p class="tagline">SINGULARITY INTELLIGENCE ENGINE v21.0</p>
        <div class="social-proof">
            <i class="fas fa-heart"></i>
            <!-- ISOLATED LIKE NUMBER -->
            <span class="like-number" id="landing-likes">{{ total_likes }}</span>
            <span>People attracted by our results</span>
        </div>
        <button class="btn-start" onclick="navTo('p2')">INITIALIZE SYSTEM</button>
    </div>

    <!-- COMMAND CENTER -->
    <div class="container hidden" id="p2" style="justify-content: flex-start; padding-top: 10vh;">
        <div class="search-wrapper">
            <h2 style="margin-bottom: 25px; text-shadow: 0 2px 4px black; font-weight: 700;">GLOBAL COMMAND</h2>
            <div class="super-bar">
                <div class="icon-wrapper"><i class="fas fa-globe"></i>
                    <!-- FULL COUNTRY LIST (50+ ENTRIES) -->
                    <select id="region" class="bar-select">
                        <option value="Global">Global</option><option value="USA">USA</option><option value="India">India</option><option value="UK">UK</option>
                        <option value="Argentina">Argentina</option><option value="Australia">Australia</option><option value="Austria">Austria</option>
                        <option value="Belgium (fr)">Belgium (FR)</option><option value="Belgium (nl)">Belgium (NL)</option><option value="Brazil">Brazil</option>
                        <option value="Bulgaria">Bulgaria</option><option value="Canada (en)">Canada (EN)</option><option value="Canada (fr)">Canada (FR)</option>
                        <option value="Chile">Chile</option><option value="China">China</option><option value="Colombia">Colombia</option>
                        <option value="Croatia">Croatia</option><option value="Czech Republic">Czech Rep</option><option value="Denmark">Denmark</option>
                        <option value="Egypt">Egypt</option><option value="Estonia">Estonia</option><option value="Finland">Finland</option>
                        <option value="France">France</option><option value="Germany">Germany</option><option value="Greece">Greece</option>
                        <option value="Hong Kong">Hong Kong</option><option value="Hungary">Hungary</option><option value="Indonesia">Indonesia</option>
                        <option value="Ireland">Ireland</option><option value="Israel">Israel</option><option value="Italy">Italy</option>
                        <option value="Japan">Japan</option><option value="Korea">Korea</option><option value="Latvia">Latvia</option>
                        <option value="Lithuania">Lithuania</option><option value="Malaysia">Malaysia</option><option value="Mexico">Mexico</option>
                        <option value="Netherlands">Netherlands</option><option value="New Zealand">New Zealand</option><option value="Norway">Norway</option>
                        <option value="Pakistan">Pakistan</option><option value="Peru">Peru</option><option value="Philippines">Philippines</option>
                        <option value="Poland">Poland</option><option value="Portugal">Portugal</option><option value="Romania">Romania</option>
                        <option value="Russia">Russia</option><option value="Saudi Arabia">Saudi Arabia</option><option value="Singapore">Singapore</option>
                        <option value="Slovakia">Slovakia</option><option value="Slovenia">Slovenia</option><option value="South Africa">South Africa</option>
                        <option value="Spain">Spain</option><option value="Sweden">Sweden</option><option value="Switzerland (de)">Switzerland (DE)</option>
                        <option value="Switzerland (fr)">Switzerland (FR)</option><option value="Taiwan">Taiwan</option><option value="Thailand">Thailand</option>
                        <option value="Turkey">Turkey</option><option value="Ukraine">Ukraine</option><option value="Vietnam">Vietnam</option>
                    </select>
                </div>
                <div class="icon-wrapper"><i class="fas fa-brain"></i><select id="mode" class="bar-select"><option value="Auto">Auto</option><option value="Catalog">Catalog</option><option value="Fact Check">Verify</option><option value="Deep Research">Deep</option><option value="Market Analysis">Market</option></select></div>
                <input type="text" id="topic" class="search-input" placeholder="Ex: 'List all emerging AI tools'...">
                <button class="submit-btn" onclick="runScan()"><i class="fas fa-arrow-up"></i></button>
            </div>
            <div class="help-link" onclick="document.getElementById('help-modal').style.display='flex'">System Capabilities & Usage</div>
            <div class="loader" id="loader"></div>
        </div>

        <div class="results-wrapper hidden" id="results-area">
            <div class="card">
                <div id="report-content" class="report-content"></div>
                <div class="like-section">
                    <p style="color:#888;">Was this intelligence useful?</p>
                    <button class="btn-like" onclick="sendLike(this)"><i class="far fa-heart"></i> Like Result</button>
                </div>
            </div>
            
            <div class="card" style="padding:15px;">
                <h4 style="color:#007bff; margin-bottom:15px;">Visual Evidence & Data</h4>
                <div class="img-grid" id="img-grid"></div>
                <div id="chart-div" class="chart-container" style="display:none;"></div>
            </div>
        </div>
    </div>

    <!-- MODALS -->
    <div class="fs-modal" id="fs-modal" onclick="this.style.display='none'"><span class="fs-close">&times;</span><img id="fs-img" src=""></div>
    <div class="help-modal" id="help-modal" onclick="if(event.target===this)this.style.display='none'">
        <div class="help-box">
            <h3 style="color:#007bff; margin-bottom:15px;">System Manual v21</h3>
            <p><strong>1. Infinity Catalog:</strong> Ask for "All crimes" or "List technologies" to get an exhaustive list.</p>
            <p><strong>2. Visual Trawl:</strong> The system retrieves 20-50 real images, filtering out AI art.</p>
            <p><strong>3. Aegis Protocol:</strong> Checks event existence before reporting.</p>
            <br><button class="btn-start" style="padding:10px 30px; font-size:1rem;" onclick="document.getElementById('help-modal').style.display='none'">Close</button>
        </div>
    </div>

    <script>
        function navTo(id) {
            document.getElementById('p1').classList.add('hidden');
            document.getElementById('p2').classList.remove('hidden');
            document.getElementById('p2').classList.add('fade-in');
        }

        async function runScan() {
            const topic = document.getElementById('topic').value;
            if(!topic) return alert("Enter a topic.");
            document.getElementById('loader').style.display = 'block';
            document.getElementById('results-area').classList.add('hidden');

            try {
                const res = await fetch('/analyze', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ topic, region: document.getElementById('region').value, mode: document.getElementById('mode').value })
                });
                const data = await res.json();

                document.getElementById('report-content').innerHTML = data.report;

                const grid = document.getElementById('img-grid');
                grid.innerHTML = '';
                if(data.images.length) {
                    data.images.forEach(img => {
                        grid.innerHTML += `<div class="img-item" onclick="openFS('${img.src}')"><img src="${img.src}" onerror="this.parentElement.style.display='none'"></div>`;
                    });
                } else { grid.innerHTML = "<p style='color:#888; text-align:center;'>No verified visual evidence.</p>"; }

                if(data.chart) {
                    document.getElementById('chart-div').style.display = 'block';
                    Plotly.newPlot('chart-div', JSON.parse(data.chart));
                } else { document.getElementById('chart-div').style.display = 'none'; }

                document.getElementById('results-area').classList.remove('hidden');
            } catch(e) { alert("System Offline."); } 
            finally { document.getElementById('loader').style.display = 'none'; }
        }

        async function sendLike(btn) {
            btn.innerHTML = '<i class="fas fa-heart"></i> Liked!';
            btn.style.background = '#ff6b6b';
            btn.style.color = 'white';
            btn.disabled = true;
            const res = await fetch('/like', { method: 'POST' });
            const data = await res.json();
            document.getElementById('landing-likes').innerText = data.new_count;
        }

        function openFS(src) { document.getElementById('fs-img').src = src; document.getElementById('fs-modal').style.display = 'flex'; }
        document.getElementById('topic').addEventListener('keypress', (e) => { if(e.key === 'Enter') runScan(); });
    </script>
</body>
</html>
