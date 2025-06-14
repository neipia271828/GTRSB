import requests
import time
from concurrent.futures import ThreadPoolExecutor
import statistics

def test_endpoint(url):
    start_time = time.time()
    try:
        response = requests.get(url)
        end_time = time.time()
        return {
            'status_code': response.status_code,
            'response_time': end_time - start_time,
            'url': url
        }
    except Exception as e:
        return {
            'status_code': None,
            'response_time': None,
            'error': str(e),
            'url': url
        }

def run_performance_test(base_url, endpoints, num_requests=10):
    results = []
    urls = [f"{base_url}{endpoint}" for endpoint in endpoints]
    
    print(f"\n=== パフォーマンステスト開始 ===")
    print(f"ベースURL: {base_url}")
    print(f"テスト対象エンドポイント: {endpoints}")
    print(f"各エンドポイントのリクエスト数: {num_requests}\n")
    
    for url in urls:
        print(f"\nテスト中: {url}")
        with ThreadPoolExecutor(max_workers=5) as executor:
            endpoint_results = list(executor.map(lambda _: test_endpoint(url), range(num_requests)))
        
        successful_requests = [r for r in endpoint_results if r['status_code'] == 200]
        if successful_requests:
            response_times = [r['response_time'] for r in successful_requests]
            avg_time = statistics.mean(response_times)
            min_time = min(response_times)
            max_time = max(response_times)
            
            print(f"成功リクエスト数: {len(successful_requests)}/{num_requests}")
            print(f"平均応答時間: {avg_time:.3f}秒")
            print(f"最小応答時間: {min_time:.3f}秒")
            print(f"最大応答時間: {max_time:.3f}秒")
        else:
            print("すべてのリクエストが失敗しました")
        
        results.extend(endpoint_results)
    
    return results

if __name__ == "__main__":
    BASE_URL = "http://133.125.84.34:8000"
    ENDPOINTS = [
        "/",  # トップページ
        "/login",  # ログインページ
        "/register",  # 登録ページ
        "/scoreboard",  # スコアボード
    ]
    
    results = run_performance_test(BASE_URL, ENDPOINTS) 