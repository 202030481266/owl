import json
from http.server import SimpleHTTPRequestHandler, HTTPServer
import os


class CustomMarkdownHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, md_directory=None, **kwargs):
        self.md_directory = md_directory
        super().__init__(*args, **kwargs)

    def do_GET(self):
        # 处理文件列表请求
        if self.path == '/file-list.json':
            file_list = []
            for md_file in os.listdir(self.md_directory):
                if md_file.endswith('.md'):
                    file_list.append({
                        'name': md_file,
                        'path': f'/get_markdown?file={md_file}'
                    })

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(file_list, ensure_ascii=False).encode('utf-8'))

        # 处理Markdown文件请求
        elif self.path.startswith('/get_markdown'):
            from urllib.parse import parse_qs, urlparse
            query = parse_qs(urlparse(self.path).query)
            filename = query.get('file', [''])[0]

            if not filename:
                self.send_error(400, "Missing file parameter")
                return

            filepath = os.path.join(self.md_directory, filename)

            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                self.send_response(200)
                self.send_header('Content-type', 'text/plain; charset=utf-8')
                self.end_headers()
                self.wfile.write(content.encode('utf-8'))
            else:
                self.send_error(404, f"File not found: {filename}")

        # 默认处理静态文件
        else:
            super().do_GET()


def start_server(md_directory, port=8000):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 创建自定义的Handler工厂函数
    def handler(*args, **kwargs):
        return CustomMarkdownHandler(*args, md_directory=md_directory, **kwargs)

    server_address = ('', port)
    httpd = HTTPServer(server_address, handler)
    print(f"服务器启动在 http://localhost:{port}")
    httpd.serve_forever()