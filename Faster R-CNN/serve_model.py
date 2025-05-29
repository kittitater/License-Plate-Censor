from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DummyInput(BaseModel):
    text: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: DummyInput):
    return {
        "prediction": "dummy_label",
        "confidence": 0.123,
        "received": data.dict()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve_model:app", host="0.0.0.0", port=80)
