import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, HttpUrl
from src.video_agent import VideoAgentPipeline
import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from pathlib import Path

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://34.123.184.43"],  # Your React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Jinja2 environment for email templates
env = Environment(
    loader=FileSystemLoader('templates')
)


class MovieClick(BaseModel):
    title: str
    email: EmailStr
    youtubeUrl: HttpUrl


class MovieResponse(BaseModel):
    message: str
    email: EmailStr
    youtubeUrl: str
    status: str


async def send_email(to_email: str, movie_title: str, download_url: str):
    """Send email with download URL to user."""
    try:
        # Get email credentials from environment variables
        smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "465"))
        sender_email = os.getenv("SENDER_EMAIL")
        sender_password = os.getenv("SENDER_PASSWORD")

        if not all([sender_email, sender_password]):
            raise ValueError("Email credentials not properly configured")

        # Create message
        msg = MIMEMultipart('alternative')
        msg['From'] = sender_email
        msg['To'] = to_email
        msg['Subject'] = f"Your Movie Recap for {movie_title} is Ready!"

        # Create HTML content
        template = env.get_template('email_template.html')
        html_content = template.render(
            movie_title=movie_title,
            download_url=download_url
        )

        # Attach both plain text and HTML versions
        text_content = f"""
        Hello,

        Your movie recap for "{movie_title}" is ready!

        You can download your video recap here: {download_url}

        Best regards,
        Your Movie Recap Team
        """
        
        msg.attach(MIMEText(text_content, 'plain'))
        msg.attach(MIMEText(html_content, 'html'))

        # Send email using aiosmtplib
        async with aiosmtplib.SMTP(
            hostname=smtp_server,
            port=smtp_port,
            use_tls=True
        ) as smtp:
            await smtp.login(sender_email, sender_password)
            await smtp.send_message(msg)

        logger.info(f"[INFO] Email sent successfully to {to_email}")
        return True
    except Exception as e:
        logger.error(f"[ERROR] Failed to send email: {str(e)}")
        return False


@app.post("/api/movie-clicked", response_model=MovieResponse)
async def movie_clicked(movie: MovieClick):
    try:
        logger.info(f"Received movie click: {movie.title}")
        logger.info(f"User email: {movie.email}")
        logger.info(f"YouTube URL: {movie.youtubeUrl}")

        BASE_DIR = Path(__file__).resolve().parent.parent  # Go up one level from src to backend
        BGM_PATH = BASE_DIR / "templates" / "audio" / "Else - Paris.mp3"
        # Initialize the VideoAgentPipeline
        pipeline = VideoAgentPipeline(
            movie_name=movie.title,
            bgm_path=BGM_PATH,
            debug_dir=None  # You can set a debug directory if needed
        )

        # Run the pipeline and get the download URL
        download_url = await pipeline.run(str(movie.youtubeUrl))

        # Send email with download URL
        email_sent = await send_email(movie.email, movie.title, download_url)
        await pipeline.genai_client.delete_all_files(pipeline.files_tracker)
        
        if not email_sent:
            raise HTTPException(
                status_code=500,
                detail="Failed to send email with download URL"
            )

        return MovieResponse(
            message=f"Successfully processed movie: {movie.title} and sent download link to your email",
            email=movie.email,
            youtubeUrl=str(movie.youtubeUrl),
            status="completed"
        )
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 