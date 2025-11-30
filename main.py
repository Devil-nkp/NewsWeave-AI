import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from datetime import datetime
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS 
import re

# ==========================================
#  CONFIGURATION (SERVER SIDE)
# ==========================================

INTERNAL_API_KEY = os.environ.get("GROQ_API_KEY")

app = FastAPI()

# Setup Templates
templates = Jinja2Templates(directory="templates")

# Input Model (Frontend only sends the topic now)
class SearchRequest(BaseModel):
    topic: str

# ==========================================
# OMNI-PRO INTELLIGENCE AGENT
# ==========================================

class OmniProAgent:
    def __init__(self):
        # use of the internal key here.
        self.llm = ChatGroq(temperature=0.1, model_name="llama-3.3-70b-versatile", api_key=INTERNAL_API_KEY)
        self.date_str = datetime.now().strftime("%B %d, %Y")

    def _smart_image_search(self, topic):
        """Retrieves REAL journalistic images."""
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
        """Multi-Vector Search (News + Data)."""
        vault = ""
        try:
            with DDGS() as ddgs:
                # News Vector
                news = list(ddgs.news(topic, max_results=4))
                for r in news:
                    vault += f"SOURCE: {r['title']} ({r['date']})\nURL: {r['url']}\nCONTENT: {r['body']}\n\n"
                
                # Data Vector
                data = list(ddgs.text(f"{topic} statistics data numbers", max_results=3))
                for r in data:
                    vault += f"SOURCE: {r['title']}\nCONTENT: {r['body']}\n\n"
        except:
            pass
        return vault

    def run_mission(self, topic):
        # 1. SEARCH
        context = self._execute_deep_search(topic)
        
        if not context:
            # Fallback to Wikipedia if web search fails
            try:
                import wikipedia
                context = wikipedia.summary(topic, sentences=5)
            except:
                return "⚠️ No verifiable data found on this topic. Please try a more specific query.", []

        # 2. SYNTHESIZE
        prompt = f"""
        You are NewsWeave Omni-Pro.
        TOPIC: {topic}
        DATE: {self.date_str}
        
        DATA VAULT:
        {context}
        
        INSTRUCTIONS:
        1. Write a Forensic Intelligence Report (HTML format).
        2. Use <h3> for headers, <p> for paragraphs, and <ul>/<li> for lists.
        3. **Verification:** If the event is fake, debunk it immediately.
        4. **Structure:**
           - **Executive Verdict**
           - **Deep Analysis**
           - **Strategic Implications**
        """
        
        try:
            report_content = self.llm.invoke(prompt).content
        except Exception as e:
            report_content = f"Error generating report: {str(e)}"
        
        # 3. IMAGES
        images = self._smart_image_search(topic)
        
        return report_content, images

# Initialize Agent Instance
agent = OmniProAgent()

# ==========================================
#  API ROUTES
# ==========================================

@app.get("/")
async def serve_interface(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_topic(request: SearchRequest):
    topic = request.topic
    print(f"🚀 Processing: {topic}")
    
    report, images = agent.run_mission(topic)
    
    return JSONResponse(content={
        "topic": topic,
        "report": report,
        "images": images
    })

# Entry point for local testing
if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)
