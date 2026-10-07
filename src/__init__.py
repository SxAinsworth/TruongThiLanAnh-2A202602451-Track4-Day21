"""Code tự viết cho topic A. Chạy từ gốc repo bằng `python -m src.<tên_script>`."""
import sys

# Console / pipe trên Windows mặc định cp1252, không in được tiếng Việt (kể cả --help).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
