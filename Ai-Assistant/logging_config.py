import logging

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Handler 1 — ke file
    file_handler = logging.FileHandler("Ai.log")
    file_handler.setLevel(logging.DEBUG)

    # Handler 2 — ke terminal
    terminal_handler = logging.StreamHandler()
    terminal_handler.setLevel(logging.WARNING)
    file_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    file_handler.setFormatter(file_fmt)
    stream_fmt = logging.Formatter("%(levelname)s |  %(message)s")
    terminal_handler.setFormatter(stream_fmt)

    logger.addHandler(file_handler)
    logger.addHandler(terminal_handler)
    return logger
