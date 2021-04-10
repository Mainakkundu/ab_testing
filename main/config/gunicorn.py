"""gunicorn WSGI server configuration."""
from multiprocessing import cpu_count
from os import environ


def max_workers():
    return 2*cpu_count() + 1


bind = '0.0.0.0:' + environ.get('PORT', '12500')
max_requests = 1000
worker_class = 'gevent'
workers = max_workers()
