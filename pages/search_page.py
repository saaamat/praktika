from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select

class SearchPage:
    def __init__(self, driver):
        self.driver = driver

    def open(self, url):
        self.driver.get(url)

    def fill_name(self, name):
        self.driver.find_element(By.XPATH, '//input[@placeholder="Имя"]').clear()
        self.driver.find_element(By.XPATH, '//input[@placeholder="Имя"]').send_keys(name)

    def fill_city(self, city):
        self.driver.find_element(By.XPATH, '//input[@placeholder="Город"]').clear()
        self.driver.find_element(By.XPATH, '//input[@placeholder="Город"]').send_keys(city)

    def fill_age(self, age_from, age_to):
        self.driver.find_element(By.XPATH, '//input[@placeholder="от 14"]').clear()
        self.driver.find_element(By.XPATH, '//input[@placeholder="от 14"]').send_keys(age_from)
        self.driver.find_element(By.XPATH, '//input[@placeholder="до 100"]').clear()
        self.driver.find_element(By.XPATH, '//input[@placeholder="до 100"]').send_keys(age_to)

    def select_gender(self, gender):
        select = Select(self.driver.find_element(By.TAG_NAME, 'select'))
        select.select_by_visible_text(gender)

    def set_weights(self, weights):
        for label, value in weights.items():
            try:
                slider_xpath = f'//div[label[text()="{label}"]]/input[@type="range"]'
                slider = self.driver.find_element(By.XPATH, slider_xpath)
                self.driver.execute_script("arguments[0].value = arguments[1]; arguments[0].dispatchEvent(new Event('input'));", slider, value)
            except:
                continue

    def click_search(self):
        self.driver.find_element(By.XPATH, '//button[text()="Поиск"]').click()

    def get_results(self):
        return self.driver.find_elements(By.CLASS_NAME, "result-card")
