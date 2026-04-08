import logging
import os
import zipfile
from datetime import datetime, timezone, timedelta
from logging.handlers import RotatingFileHandler

_TZ7 = timezone(timedelta(hours=7))
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class ZipRotatingFileHandler(RotatingFileHandler):
    def doRollover(self) -> None:
        if self.stream:
            self.stream.flush()
            self.stream.close()
            self.stream = None  # type: ignore[assignment]

        now_str = datetime.now(tz=_TZ7).strftime("%Y%m%d_%H%M%S")
        zip_path = f"{self.baseFilename}.{now_str}.zip"
        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(self.baseFilename, arcname=os.path.basename(self.baseFilename))
            os.remove(self.baseFilename)
        except Exception:
            pass
        self.stream = self._open()


def setup_logging(log_cfg: dict) -> None:
    os.makedirs(log_cfg["log_dir"], exist_ok=True)
    log_path = os.path.join(log_cfg["log_dir"], "app.log")

    level = getattr(logging, log_cfg["level"].upper(), logging.INFO)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(level)

    fh = ZipRotatingFileHandler(
        filename=log_path, maxBytes=log_cfg["max_bytes"], backupCount=0, encoding="utf-8",
    )
    fh.setFormatter(formatter)
    fh.setLevel(level)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(fh)

    if level > logging.DEBUG:
        for noisy in ("sentence_transformers", "transformers", "faiss", "torch", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
