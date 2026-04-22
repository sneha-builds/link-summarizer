# AI Link Summarizer 

A full-stack web application that extracts text from any provided URL and generates a concise summary using Natural Language Processing (NLP).

##  Features
- **Web Scraping:** Automatically extracts article content using BeautifulSoup4.
- **AI-Powered:** Uses the `distilbart-cnn-12-6` Transformer model for high-quality abstractive summarization.
- **Modern Backend:** Built with FastAPI (Python 3.13+) for high-performance asynchronous API handling.
- **Clean UI:** A minimalist, responsive frontend for easy user interaction.

##  Tech Stack
- **Frontend:** HTML5, CSS3, JavaScript (Fetch API)
- **Backend:** [FastAPI](https://fastapi.tiangolo.com/)
- **Machine Learning:** [Hugging Face Transformers](https://huggingface.co/docs/transformers/index), PyTorch
- **Scraping:** Requests, BeautifulSoup4

##  Getting Started

### Prerequisites
- Python 3.13 or higher
- `pip` (Python package manager)

### Installation
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/link-summarizer.git](https://github.com/yourusername/link-summarizer.git)
   cd link-summarizer

2. Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    Install dependencies:

3. Install dependencies:

    ```bash
    pip install fastapi uvicorn requests beautifulsoup4 transformers torch sentencepiece protobuf

## Running the Application

1. Start the server:

    ```bash
    uvicorn main:app --reload
   
2. Access the UI:

Open your browser and navigate to http://127.0.0.1:8000.

4. First Run Note: The application will download the AI model (~1.2GB) on the first startup. Please wait until you see "Model loaded successfully!" in the terminal.

## How it Works

Request: The user submits a URL via the web interface.

Extraction: The backend fetches the HTML, filters for paragraph tags (<p>), and cleans the text.

Processing: The text is tokenized and passed through a Sequence-to-Sequence (Seq2Seq) model.

Result: The generated summary is sent back as a JSON response and displayed on the UI.

## Configuration
Input Limit: Currently set to process the first 2,000 characters of an article to optimize performance.

Model: sshleifer/distilbart-cnn-12-6 (Distilled version of BART for speed).

🤝 Contributing

Feel free to fork this project, report issues, or submit pull requests to improve the summarization logic or UI.
