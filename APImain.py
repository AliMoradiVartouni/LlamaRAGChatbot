# from basicRAG import responsechat
# from advanceRAG import responsechat
from RAGPlus import responsechat
import os
from time import perf_counter

from fastapi import FastAPI, Request, HTTPException
import uvicorn
from pydantic import BaseModel

app = FastAPI()


# Define request and response models
class UserInput(BaseModel):
    message: str

class BotResponse(BaseModel):
    response: str


# Initialize the bot
model_dir = "/home/ali/moradi/models/Radman-Llama-3.2-3B/extra"
if not os.path.exists(model_dir):
    raise ValueError(f"Model directory not found: {model_dir}")

# Initialize chat instance with model directory
bot = responsechat(model_dir)

@app.get("/")
def read_root():
    return {"message": "Bot is ready! Type 'quit' to exit. (Note: Use POST /chat for interaction)"}


@app.post("/chat", response_model=BotResponse)
async def main(user_input: UserInput):
    try:
        print(f"Received message: {user_input.message}")  # Debug print
        # Get the bot response
        result = bot.get_response(user_input.message)
        print(f"Bot response: {result}")  # Debug print
        return BotResponse(response = result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# uvicorn FairFace:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
