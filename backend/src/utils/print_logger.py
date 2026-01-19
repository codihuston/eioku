"""Print-based logger adapter for debugging when standard logging doesn't work."""


class PrintLogger:
    """Logger that uses print() for immediate output visibility."""

    def __init__(self, name: str):
        self.name = name

    def debug(self, msg: str, *args, **kwargs):
        """Log debug message."""
        formatted_msg = msg % args if args else msg
        print(f"🔍 DEBUG [{self.name}] {formatted_msg}", flush=True)

    def info(self, msg: str, *args, **kwargs):
        """Log info message."""
        formatted_msg = msg % args if args else msg
        print(f"ℹ️  INFO [{self.name}] {formatted_msg}", flush=True)

    def warning(self, msg: str, *args, **kwargs):
        """Log warning message."""
        formatted_msg = msg % args if args else msg
        print(f"⚠️  WARNING [{self.name}] {formatted_msg}", flush=True)

    def error(self, msg: str, *args, **kwargs):
        """Log error message."""
        formatted_msg = msg % args if args else msg
        print(f"❌ ERROR [{self.name}] {formatted_msg}", flush=True)

    def critical(self, msg: str, *args, **kwargs):
        """Log critical message."""
        formatted_msg = msg % args if args else msg
        print(f"🚨 CRITICAL [{self.name}] {formatted_msg}", flush=True)


def get_logger(name: str) -> PrintLogger:
    """Get a print-based logger instance."""
    return PrintLogger(name)
