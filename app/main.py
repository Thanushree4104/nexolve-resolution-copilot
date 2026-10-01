from fastapi import FastAPI

app = FastAPI(title="Nexolve Resolution Copilot")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}