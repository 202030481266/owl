import json
from http.server import SimpleHTTPRequestHandler, HTTPServer
import os
import mimetypes


class CustomMarkdownHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, md_directory=None, **kwargs):
        self.md_directory = md_directory
        super().__init__(*args, **kwargs)

    def do_GET(self):
        # 处理文件列表请求
        if self.path == '/file-list.json':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.end_headers()
            
            file_list = []
            try:
                for md_file in sorted(os.listdir(self.md_directory)):
                    if md_file.endswith('.md'):
                        # 获取文件最后修改时间
                        file_path = os.path.join(self.md_directory, md_file)
                        file_stats = os.stat(file_path)
                        last_modified = file_stats.st_mtime
                        
                        file_list.append({
                            'name': md_file,
                            'path': f'/get_markdown?file={md_file}',
                            'lastModified': last_modified
                        })
                self.wfile.write(json.dumps(file_list, ensure_ascii=False).encode('utf-8'))
            except Exception as e:
                self.wfile.write(json.dumps({"error": str(e)}, ensure_ascii=False).encode('utf-8'))

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
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()

                    self.send_response(200)
                    self.send_header('Content-type', 'text/markdown; charset=utf-8')
                    self.send_header('Cache-Control', 'max-age=60')  # 缓存1分钟
                    self.end_headers()
                    self.wfile.write(content.encode('utf-8'))
                except UnicodeDecodeError:
                    # 尝试使用其他编码
                    with open(filepath, 'r', encoding='gbk') as f:
                        content = f.read()
                    self.send_response(200)
                    self.send_header('Content-type', 'text/markdown; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(content.encode('utf-8'))
            else:
                self.send_error(404, f"File not found: {filename}")

        # 默认处理静态文件
        else:
            # 添加对主要静态文件的MIME类型支持
            if self.path.endswith('.css'):
                content_type = 'text/css'
            elif self.path.endswith('.js'):
                content_type = 'application/javascript'
            elif self.path.endswith('.html'):
                content_type = 'text/html'
            elif self.path.endswith('.json'):
                content_type = 'application/json'
            elif self.path.endswith('.png'):
                content_type = 'image/png'
            elif self.path.endswith('.jpg') or self.path.endswith('.jpeg'):
                content_type = 'image/jpeg'
            elif self.path.endswith('.svg'):
                content_type = 'image/svg+xml'
            else:
                # 使用mimetypes猜测MIME类型
                content_type = mimetypes.guess_type(self.path)[0]
            
            if content_type:
                self.send_response(200)
                self.send_header('Content-type', content_type)
                if content_type.startswith('image/'):
                    self.send_header('Cache-Control', 'max-age=86400')  # 图片缓存1天
                else:
                    self.send_header('Cache-Control', 'max-age=3600')   # 其他静态文件缓存1小时
                try:
                    super().do_GET()
                except:
                    self.send_error(404, f"File not found: {self.path}")
            else:
                super().do_GET()


def start_server(md_directory, port=8000):
    # 确保目录存在
    if not os.path.exists(md_directory):
        print(f"Warning: Directory {md_directory} does not exist. Creating it...")
        os.makedirs(md_directory)
        
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 创建自定义的Handler工厂函数
    def handler(*args, **kwargs):
        return CustomMarkdownHandler(*args, md_directory=md_directory, **kwargs)

    server_address = ('', port)
    httpd = HTTPServer(server_address, handler)
    print(f"服务器启动在 http://localhost:{port}")
    print(f"Markdown文件目录: {os.path.abspath(md_directory)}")
    httpd.serve_forever()