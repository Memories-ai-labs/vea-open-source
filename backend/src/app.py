from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MovieClick(BaseModel):
    title: str
    email: EmailStr

@app.post("/api/movie-clicked")
async def movie_clicked(movie: MovieClick):
    # Print the received movie name to console
    print(f"Received movie click: {movie.title}")
    print(f"User email: {movie.email}")
    return {
        "message": f"Received movie: {movie.title}",
        "email": movie.email
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 