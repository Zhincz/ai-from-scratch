"""Cho pytest thấy thư mục gốc (để import utils) và từng thư mục con trong
fundamentals/. Các thư mục con bắt đầu bằng số nên không import kiểu package
bình thường được, nên mình nhét thẳng vào sys.path."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
for d in (ROOT / "fundamentals").glob("*/"):
    sys.path.insert(0, str(d))
