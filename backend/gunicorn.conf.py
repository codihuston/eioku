# gunicorn.conf.py
# This file tells Gunicorn where to send its logs.
# The JSON formatting is handled by the application itself.

bind = "0.0.0.0:8000"
workers = 2
worker_class = "uvicorn.workers.UvicornWorker"

# Send Gunicorn's access and error logs to stdout and stderr
accesslog = "-"
errorlog = "-"

# Disable buffering for real-time logs
forwarded_allow_ips = "*"
