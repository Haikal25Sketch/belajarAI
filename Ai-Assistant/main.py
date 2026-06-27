import warnings
import os

# Filter warning agar tampilan bersih
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", message=".*allowed_objects.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["PYTHONWARNINGS"] = "ignore"

from logging_config import setup_logging
from agent_graph import agent

logger = setup_logging()

while True:
    user = input("\nKamu: ")
    if user.lower() == "keluar":
        break
    agent(user)
