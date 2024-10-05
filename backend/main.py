from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ollama
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
import PyPDF2
import io

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    message: str
    session_id: str
    use_pdf: bool = False

chat_sessions = {}
pdf_contents = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a ThreadPoolExecutor
executor = ThreadPoolExecutor()

@app.post("/chat")
async def chat(chat_message: ChatMessage):
    try:
        logger.info(f"Received chat message: {chat_message}")
        session_id = chat_message.session_id
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []

        full_message = chat_message.message
        if chat_message.use_pdf and session_id in pdf_contents:
            pdf_content = pdf_contents[session_id]
            full_message = f"PDF Content: {pdf_content}\n\nUser Question: {chat_message.message}"

        chat_sessions[session_id].append({
            'role': 'user',
            'content': full_message
        })

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: ollama.chat(model="llama2", messages=chat_sessions[session_id])
            )
            ai_response = response['message']['content']
        except Exception as e:
            logger.error(f"Ollama error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")
        
        logger.info(f"AI response received: {ai_response[:100]}...")  # Log 100 aksara pertama
        chat_sessions[session_id].append({
            'role': 'assistant',
            'content': ai_response
        })

        return {"response": ai_response}
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), session_id: str = Form(...)):
    logger.info(f"Received upload request for session_id: {session_id}")
    if not file:
        logger.error("No file uploaded")
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    if not session_id:
        logger.error("No session ID provided")
        raise HTTPException(status_code=400, detail="No session ID provided")
    
    try:
        content = await file.read()
        logger.info(f"File read successfully: {file.filename}")
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        text_content = ""
        for page in pdf_reader.pages:
            text_content += page.extract_text() + "\n"
        
        pdf_contents[session_id] = text_content
        logger.info(f"PDF processed and stored for session_id: {session_id}")
        
        return {
            "message": f"File processed successfully",
            "filename": file.filename
        }
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.get("/sessions")
async def get_sessions():
    return {"sessions": list(chat_sessions.keys())}

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"messages": chat_sessions[session_id]}

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        return {"message": f"Session {session_id} deleted successfully"}
    raise HTTPException(status_code=404, detail="Session not found")

@app.delete("/sessions")
async def clear_all_sessions():
    chat_sessions.clear()
    return {"message": "All chat sessions cleared"}
