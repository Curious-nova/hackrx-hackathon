# # app/config.py

# from dotenv import load_dotenv
# from pydantic_settings import BaseSettings, SettingsConfigDict

# # Forcefully load the .env file into the environment
# load_dotenv()

# class Settings(BaseSettings):
#     """Loads configuration from the .env file."""
#     model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra='ignore')

#     # API Keys
#     pinecone_api_key: str
#     together_api_key: str

#     # Hackathon Auth Token
#     hackathon_api_token: str

#     # Pinecone Config
#     pinecone_index_name: str

# settings = Settings()
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra='ignore')
    google_api_key: str
    pinecone_api_key: str
    hackathon_api_token: str
    pinecone_index_name: str

settings = Settings()