import pytest
import json
from selenium import webdriver
from pages.search_page import SearchPage
from utils.logger import logger


def load_test_data():
    with open("data/search_cases.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def driver():
    driver = webdriver.Chrome()
    yield driver
    driver.quit()


@pytest.mark.parametrize("test_case", load_test_data())
def test_user_search(driver, test_case):
    page = SearchPage(driver)
    page.open("http://localhost:8000")  # Укажи свой URL

    logger.info(f"▶️ Тест: {test_case}")

    page.fill_name(test_case["name"])
    page.fill_city(test_case["city"])
    page.fill_age(test_case["age_from"], test_case["age_to"])
    page.select_gender(test_case["gender"])
    page.set_weights(test_case["weights"])
    page.click_search()

    results = page.get_results()
    assert results, "❌ Результаты не найдены"
    logger.info(f"✅ Найдено результатов: {len(results)}")
