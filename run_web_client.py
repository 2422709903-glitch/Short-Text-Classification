"""打开本地模型对比网页。请先运行 server.py。"""

import webbrowser


if __name__ == '__main__':
    url = 'http://127.0.0.1:5000'
    print(f'正在打开 {url}')
    webbrowser.open(url)

