import logging
import sys


def setup_logging():
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Silence noisy libs
    for lib in ("httpx", "motor", "pymongo", "passlib"):
        logging.getLogger(lib).setLevel(logging.WARNING)
