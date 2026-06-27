from rag import cari_database as _cari_database
from langchain_core.tools import tool

@tool
def cuaca(query):
    """Check the weather in Jakarta, Bogor, or Bekasi."""
    database = {
        "jakarta": "10°C",
        "bogor": "13°C",
        "bekasi": "12°C"
    }
    for key in database:
        if key in query.lower():
            return database[key]
    return "Data tidak tersedia"

@tool
def kalkulator(ekspresi):
    """Evaluate a mathematical expression."""
    try:
        return str(eval(ekspresi))
    except:
        return "Ekspresi tidak valid"

@tool
def cari_database(query:str):
    """Search for specific information in a knowledge database to answer a user's question."""
    return _cari_database(query)

DAFTAR_TOOLS =[cuaca,kalkulator,cari_database]
