from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests
from bs4 import BeautifulSoup
# We use AutoTokenizer and AutoModel instead of 'pipeline' for better stability
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()

# --- 1. CORS Setup (Crucial for the Frontend to work) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. Model Initialization ---
# We define these globally so the function below can "see" them
model_name = "sshleifer/distilbart-cnn-12-6"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

print("Loading AI model... this may take a moment.")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
print("Model loaded successfully!")


class LinkRequest(BaseModel):
    url: str

# --- 3. The Summarization Logic ---
@app.post("/summarize")
async def summarize_link(request: LinkRequest):
    try:
        # Fetch the webpage
        response = requests.get(request.url, timeout=10)
        response.raise_for_status()
        
        # Extract text from HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        article_text = " ".join([p.get_text() for p in paragraphs])
        
        # Prepare the text for the model
        input_text = article_text[:2000] 
        if len(input_text) < 100:
            return {"summary": "Content too short to summarize."}

        # Step A: Convert words to numbers (Encoding)
        inputs = tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=1024, 
            truncation=True
        )
        
        # Step B: AI generates summary numbers
        summary_ids = model.generate(
            inputs["input_ids"], 
            max_length=130, 
            min_length=30, 
            num_beams=4, 
            early_stopping=True
        )
        
        # Step C: Convert numbers back to words (Decoding)
        summary_result = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return {"summary": summary_result}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))