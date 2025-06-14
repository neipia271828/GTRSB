import unittest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time

class TestFrontend(unittest.TestCase):
    def setUp(self):
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.implicitly_wait(10)

    def tearDown(self):
        self.driver.quit()

    def test_page_load(self):
        """ページが正しく読み込まれることを確認"""
        self.driver.get('http://127.0.0.1:5001')
        self.assertIn('GTRラップタイム記録アプリ', self.driver.title)

    def test_login_form_display(self):
        """ログインフォームが表示されることを確認"""
        self.driver.get('http://127.0.0.1:5001')
        login_button = self.driver.find_element(By.ID, 'loginButton')
        login_button.click()
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.ID, 'loginForm'))
        )

    def test_register_form_display(self):
        """登録フォームが表示されることを確認"""
        self.driver.get('http://127.0.0.1:5001')
        register_button = self.driver.find_element(By.ID, 'registerButton')
        register_button.click()
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.ID, 'registerForm'))
        )

if __name__ == '__main__':
    unittest.main() 