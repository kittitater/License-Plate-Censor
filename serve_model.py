from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

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
