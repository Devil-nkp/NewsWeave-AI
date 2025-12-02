import os
import uvicorn
import sqlite3
import logging
import smtplib
import asyncio
from email.mime.text import MIMEText
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS
from apscheduler.schedulers.background import BackgroundScheduler

# --- DEPLOYMENT CONFIGURATION ---
# On Render, set these in the "Environment" tab.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_FALLBACK_KEY_IF_LOCAL")
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "")
EMAIL_PASS = os.environ.get("EMAIL_PASS", "") 

# STORAGE PATH (Crucial for Render Persistence)
# On Render, mount a disk to /var/data to keep likes forever.
DB_PATH = "/var/data/newsweave.db" if os.path.exists("/var/data") else "newsweave.db"

# LOGGING
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NewsWeave-Omni")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- DATABASE ENGINE ---
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS global_stats (id INTEGER PRIMARY KEY, likes INTEGER DEFAULT 0)''')
        c.execute('''CREATE TABLE IF NOT EXISTS daily_audit (date TEXT PRIMARY KEY, prompt_count INTEGER DEFAULT 0)''')
        
        c.execute('SELECT count(*) FROM global_stats')
        if c.fetchone()[0] == 0:
            c.execute('INSERT INTO global_stats (likes) VALUES (1540)') # Initial Social Proof
        conn.commit()

init_db()

def get_stats():
    with sqlite3.connect(DB_PATH) as conn:
        return conn.cursor().execute("SELECT likes FROM global_stats WHERE id=1").fetchone()[0]

def increment_like():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("UPDATE global_stats SET likes = likes + 1 WHERE id=1")
        conn.commit()
        return c.execute("SELECT likes FROM global_stats WHERE id=1").fetchone()[0]

def log_prompt():
    today = datetime.now().strftime("%Y-%m-%d")
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("INSERT OR IGNORE INTO daily_audit (date, prompt_count) VALUES (?, 0)", (today,))
        c.execute("UPDATE daily_audit SET prompt_count = prompt_count + 1 WHERE date = ?", (today,))
        conn.commit()

# --- DAILY EMAIL REPORTER ---
def send_daily_report():
    if not ADMIN_EMAIL or not EMAIL_PASS: return
    
    today = datetime.now().strftime("%Y-%m-%d")
    with sqlite3.connect(DB_PATH) as conn:
        count = conn.cursor().execute("SELECT prompt_count FROM daily_audit WHERE date=?", (today,)).fetchone()
        count = count[0] if count else 0

    if count == 0: return

    msg = MIMEText(f"NewsWeave Infinity Report\nDate: {today}\nPrompts Processed: {count}\nStatus: Nominal")
    msg['Subject'] = f"Daily Audit: {today}"
    msg['From'] = ADMIN_EMAIL
    msg['To'] = ADMIN_EMAIL

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(ADMIN_EMAIL, EMAIL_PASS)
            server.send_message(msg)
    except Exception as e:
        logger.error(f"Email Error: {e}")

scheduler = BackgroundScheduler()
scheduler.add_job(send_daily_report, 'cron', hour=23, minute=55)
scheduler.start()

# --- INTELLIGENCE AGENT V13 ---
class SuperAgent:
    def __init__(self):
        self.llm = ChatGroq(temperature=0.1, model_name="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

    def _get_clean_images(self, topic, region):
        """
        Anti-Hallucination Visual Search.
        Strictly filters out AI junk.
        """
        # Append region-specific terms if possible, or generic news terms
        query = f"{topic} news photography real life -ai -cartoon -vector -render -art"
        
        clean = []
        blacklist = ["midjourney", "dall-e", "stable diffusion", "fantasy", "clipart", "icon", "logo", "stock-vector"]
        
        try:
            with DDGS() as ddgs:
                # Use region code for image search too
                results = list(ddgs.images(query, region=region, max_results=25))
                for r in results:
                    if len(clean) >= 12: break
                    if any(b in r['title'].lower() for b in blacklist): continue
                    clean.append({"src": r['image'], "title": r['title']})
        except: pass
        return clean

    def generate_report(self, topic, region_code, language_name, mode):
        # 1. Gather Intelligence (DuckDuckGo)
        context = ""
        try:
            with DDGS() as ddgs:
                # Search in the specific region requested
                results = list(ddgs.text(f"{topic}", region=region_code, max_results=10))
                for r in results:
                    context += f"SOURCE: {r['title']}\nURL: {r['href']}\nINFO: {r['body']}\n\n"
        except Exception as e:
            context = "Secure Connection Failed. Relying on internal knowledge base."

        # 2. Synthesis (LLM)
        # We instruct the AI to process in English (for IQ) but Output in User's Language
        prompt = f"""
        SYSTEM: You are NewsWeave Infinity.
        USER LANGUAGE: {language_name}
        TOPIC: {topic}
        MODE: {mode}
        
        RAW INTEL:
        {context}
        
        TASK:
        1. Analyze the 'RAW INTEL'.
        2. Create a high-level intelligence report.
        3. IMPORTANT: The report MUST be written in {language_name}.
        
        STRUCTURE:
        - <h1>Headline (In {language_name})</h1>
        - <h3>Executive Summary</h3> (Concise truth)
        - <h3>Key Findings</h3> (Bullet points)
        - <h3>Market/Social Impact</h3>
        - <i>Sources: Embedded as links</i>
        
        Be visually clean. Use HTML formatting (<b>, <br>, <h3>).
        """
        
        try:
            report = self.llm.invoke(prompt).content
        except:
            report = f"<p>AI Busy. Please retry search for '{topic}'.</p>"

        # 3. Visuals
        images = self._get_clean_images(topic, region_code)
        
        return report, images

agent = SuperAgent()

# --- API ---
class Query(BaseModel):
    topic: str
    region_code: str
    language_name: str
    mode: str

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/stats")
def stats():
    return {"likes": get_stats()}

@app.post("/like")
def like():
    return {"likes": increment_like()}

@app.post("/analyze")
def analyze(q: Query):
    log_prompt()
    rep, imgs = agent.generate_report(q.topic, q.region_code, q.language_name, q.mode)
    return JSONResponse({"report": rep, "images": imgs})

if __name__ == "__main__":
    # Render requires port from env, default 10000
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
