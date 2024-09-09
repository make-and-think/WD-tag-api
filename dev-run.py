import uvicorn
import sys
import os

# Добавляем корневую директорию проекта в sys.path
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.main import app

if __name__ == "__main__":
    # TODO parse host and port arguments from command line
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)