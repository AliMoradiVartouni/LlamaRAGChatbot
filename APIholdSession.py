from RAGPlus import responsechat
import os
from time import perf_counter

from fastapi import FastAPI, Request, HTTPException
import uvicorn
from pydantic import BaseModel
from collections import defaultdict
import uuid

app = FastAPI()
# Store user sessions
user_sessions = defaultdict(list)


# Define request and response models
class UserInput(BaseModel):
    message: str

class BotResponse(BaseModel):
    response: str

# In-memory session storage
sessions = {}

class StartConversationRequest(BaseModel):
    user_name: str

class SendMessageRequest(BaseModel):
    session_id: str
    message: str

# Initialize the bot
model_dir = "/home/ali/moradi/models/Radman-Llama-3.2-3B/extra"
if not os.path.exists(model_dir):
    raise ValueError(f"Model directory not found: {model_dir}")

# Initialize chat instance with model directory
bot = responsechat(model_dir)

@app.get("/")
def read_root():
    return {"message": "Bot is ready! Type 'quit' to exit. (Note: Use POST /chat for interaction)"}

@app.post("/start_conversation", response_model=dict)
def start_conversation(request: StartConversationRequest):
    session_id = str(uuid.uuid4())
    user_name = request.user_name
    sessions[session_id] = {
        "user_name": user_name,
        "context": []
    }
    return {"session_id": session_id}
@app.post("/chat", response_model=BotResponse)
async def main(user_input: SendMessageRequest):
    session_id = user_input.session_id
    message = user_input.message

    if not session_id or not message:
        raise HTTPException(status_code=400, detail="Missing session_id or message")

    # Retrieve session from in-memory storage
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Update context
    session["context"].append(message)

    # Get the bot response
    try:
        print(f"Received message: {message}")  # Debug print
        result = bot.get_response(message)
        print(f"Bot response: {result}")  # Debug print
        session["context"].append(result)  # Optionally, store the bot's response in the context
        return BotResponse(response=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# uvicorn FairFace:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)
