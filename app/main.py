from fastapi import FastAPI

app = FastAPI()


@app.on_event("startup")
def on_startup():
    # load classes in memory
    pass
