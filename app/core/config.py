from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    embedding_model: str = "all-MiniLM-L6-v2"
    vector_db: str = "chroma"
    persist_directory: str = "./vectorstore"
    class Config:
        env_file = ".env"

settings = Settings()
