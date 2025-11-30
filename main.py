import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles # Import this
from pydantic import BaseModel
from datetime import datetime
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS 

# --- CONFIGURATION ---
# Uses Render Environment Variable or defaults to your key
INTERNAL_API_KEY = os.environ.get("GROQ_API_KEY")

app = FastAPI()

# 1. MOUNT STATIC FILES (Crucial for local video)
# This tells the server: "Allow access to files in the 'static' folder"
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

class SearchRequest(BaseModel):
    topic: str

# ==========================================
# 🧠 OMNI-PRO AGENT LOGIC
# ==========================================

class OmniProAgent:
    def __init__(self):
        self.llm = ChatGroq(temperature=0.1, model_name="llama-3.3-70b-versatile", api_key=INTERNAL_API_KEY)
        self.date_str = datetime.now().strftime("%B %d, %Y")

    def _smart_image_search(self, topic):
        search_query = f"{topic} news photo event"
        gallery = []
        try:
            with DDGS() as ddgs:
                results = list(ddgs.images(search_query, max_results=8))
                for r in results:
                    title = r.get('title', '').lower()
                    if "ai generated" not in title and "cartoon" not in title:
                        gallery.append({"src": r['image'], "title": r['title']})
                        if len(gallery) >= 4: break
        except:
            pass
        return gallery

    def _execute_deep_search(self, topic):
        vault = ""
        try:
            with DDGS() as ddgs:
                news = list(ddgs.news(topic, max_results=4))
                for r in news:
                    vault += f"SOURCE: {r['title']} ({r['date']})\nURL: {r['url']}\nCONTENT: {r['body']}\n\n"
                
                data = list(ddgs.text(f"{topic} statistics data numbers", max_results=3))
                for r in data:
                    vault += f"SOURCE: {r['title']}\nCONTENT: {r['body']}\n\n"
        except:
            pass
        return vault

    def run_mission(self, topic):
        context = self._execute_deep_search(topic)
        if not context:
            # Simple Wikipedia fallback if search blocks
            try:
                import wikipedia
                context = wikipedia.summary(topic, sentences=4)
            except:
                return "⚠️ No verifiable data found.", []

        prompt = f"""
        You are NewsWeave Omni-Pro.
        TOPIC: {topic}
        DATE: {self.date_str}
        DATA VAULT: {context}
        INSTRUCTIONS: Write a Forensic Intelligence Report (HTML format) with Executive Verdict, Deep Analysis, and Strategic Implications.
        """
        try:
            report_content = self.llm.invoke(prompt).content
        except Exception as e:
            report_content = f"Error: {str(e)}"
        
        images = self._smart_image_search(topic)
        return report_content, images

agent = OmniProAgent()

@app.get("/")
async def serve_interface(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_topic(request: SearchRequest):
    report, images = agent.run_mission(request.topic)
    return JSONResponse(content={"topic": request.topic, "report": report, "images": images})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
