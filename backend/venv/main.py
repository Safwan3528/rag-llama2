from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pypdf
import ollama

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # Read PDF content
    pdf_content = await file.read()
    pdf_reader = pypdf.PdfReader(BytesIO(pdf_content))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Process with Ollama
    response = ollama.generate(model="llama2", prompt=f"Summarize the following text:\n\n{text}")

    return {"summary": response['response']}

@app.post("/analyze")
async def analyze_text(text: str):
    response = ollama.generate(model="llama2", prompt=f"Analyze the following text:\n\n{text}")
    return {"analysis": response['response']}
