import os
from dotenv import load_dotenv

load_dotenv()


def getProjectName():
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    return PROJECT_ID
