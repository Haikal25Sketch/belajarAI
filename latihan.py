import os
import warnings
from dotenv import load_dotenv

# Membungkam peringatan Deprecation agar tidak mengganggu di terminal

warnings.filterwarnings("ignore", category=DeprecationWarning)
try:
    from langchain_core._api import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except ImportError:
    pass

from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory



prompt = ChatPromptTemplate.from_messages,MessagesPlaceholder
for i in prompt:
    print (i)
