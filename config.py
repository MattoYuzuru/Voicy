import os

import dotenv

dotenv.load_dotenv()


KEY = os.getenv("GIGACHAT_API_KEY")
TOKEN = os.getenv("BOT_TOKEN")