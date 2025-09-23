from celery import Celery
import os

# Celery 的消息代理(broker)和结果后端(backend)都使用Redis
# 确保你的 Redis 服务正在运行
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

celery_app = Celery(
    'tasks',
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=['web_app.tasks'] # 告诉 Celery 去哪里找任务定义
)

if __name__ == '__main__':
    celery_app.start()