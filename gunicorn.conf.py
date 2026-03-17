# gunicorn.conf.py
import os

# 绑定端口（Render 会自动分配 $PORT 环境变量）
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"

# 工作进程数（推荐为 CPU 核心数 * 2 + 1，免费实例用 2 足够）
workers = 2

# 线程数（每个 worker 的线程数）
threads = 4

# 超时时间（秒），防止长时间无响应的请求挂住服务
timeout = 120

# 日志级别
loglevel = "info"

# 访问日志和错误日志（Render 会自动收集，这里可以留空）
accesslog = "-"
errorlog = "-"