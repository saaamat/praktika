import logging

logger = logging.getLogger("search_tests")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("search_test.log", mode="w", encoding="utf-8")
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
