import unittest
from app import app, db, User, Lap
import os
import tempfile

class TestApp(unittest.TestCase):
    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp()
        app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{self.db_path}'
        app.config['TESTING'] = True
        self.app = app.test_client()
        with app.app_context():
            db.create_all()

    def tearDown(self):
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def test_register(self):
        # 正常な登録
        response = self.app.post('/register', data={
            'email': 'kmc1234@kamiyama.ac.jp',
            'password': '1234',
            'username': 'testuser'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        
        # 不正なメールアドレス
        response = self.app.post('/register', data={
            'email': 'invalid@example.com',
            'password': '1234',
            'username': 'testuser2'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        
        # 不正なパスワード
        response = self.app.post('/register', data={
            'email': 'kmc5678@kamiyama.ac.jp',
            'password': '12345',
            'username': 'testuser3'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)

    def test_login(self):
        # ユーザー登録
        self.app.post('/register', data={
            'email': 'kmc1234@kamiyama.ac.jp',
            'password': '1234',
            'username': 'testuser'
        })
        
        # 正常なログイン
        response = self.app.post('/login', data={
            'email': 'kmc1234@kamiyama.ac.jp',
            'password': '1234'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        
        # 不正なパスワード
        response = self.app.post('/login', data={
            'email': 'kmc1234@kamiyama.ac.jp',
            'password': '5678'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)

    def test_add_lap(self):
        # ユーザー登録とログイン
        self.app.post('/register', data={
            'email': 'kmc1234@kamiyama.ac.jp',
            'password': '1234',
            'username': 'testuser'
        })
        self.app.post('/login', data={
            'email': 'kmc1234@kamiyama.ac.jp',
            'password': '1234'
        })
        
        # ラップタイム記録
        response = self.app.post('/add_lap', data={
            'game_title': 'Gran Turismo 7',
            'car_model': 'Nissan GT-R',
            'track_name': 'Suzuka Circuit',
            'lap_time': '90.500',
            'notes': 'テスト記録'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main() 