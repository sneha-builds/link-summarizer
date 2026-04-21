from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

app = FastAPI()

# Load the summarization model (this might take a minute on first run)
# We use 'distilbart' because it's fast and lightweight

try:
    summarizer = pipeline(
        "text2text-generation", 
        model="sshleifer/distilbart-cnn-12-6", 
        framework="pt"
    )
    print("AI model loaded successfully!")
except Exception as e:
    print("Failed to load model: {e}")

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

class LinkRequest(BaseModel):
    url: str

@app.post("/summarize")
async def summarize_link(request: LinkRequest):
    try:
        # 1. Fetch the webpage
        response = requests.get(request.url, timeout=10)
        response.raise_for_status()
        
        # 2. Parse HTML and extract text
        soup = BeautifulSoup(response.text, 'html.parser')
        # We target paragraphs to avoid navigation menus/footers
        paragraphs = soup.find_all('p')
        article_text = " ".join([p.get_text() for p in paragraphs])
        
        # Limit text length (models have a maximum token limit)
        input_text = article_text[:2000] 
        
        if len(input_text) < 100:
            return {"summary": "Content too short to summarize."}

        # 3. Summarize
        summary = summarizer(input_text, max_length=130, min_length=30, do_sample=False)
        
        return {"summary": summary[0]['summary_text']}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Boilerplate to allow frontend to talk to backend (CORS)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)