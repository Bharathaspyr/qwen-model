from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "Qwen/Qwen2.5-3B-Instruct"

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
except Exception as e:
    raise RuntimeError(f"Model or tokenizer loading failed: {str(e)}")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any domain (change if needed)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Define request schema
class InputText(BaseModel):
    text: str

@app.get("/")
async def welcome_user():
    return {"message": "Welcome to the Medical Caption Enhancement API!"}

@app.post("/generate")
async def generate_text(input_data: InputText):
    try:
        Prepare refined prompt for Qwen
        messages = [
            {"role": "system", "content": "You are a medical AI that enhances radiology reports with clarity and completeness."},
            {"role": "user", "content": f"Given this radiology observation:\n\n{input_data.text}\n\nProvide a more detailed, structured, and comprehensive report with medical insights."}
        ]

        # Use chat template if available, otherwise construct manually
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = f"System: You are a medical AI that enhances radiology reports with clarity and completeness.\nUser: Given this radiology observation:\n\n{input_data.text}\n\nProvide a more detailed, structured, and comprehensive report with medical insights.\nAI:"

        # Tokenize input
        model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(**model_inputs, max_new_tokens=512)

        # Decode response
        response_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return {"response": response_text.strip()}
   
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
