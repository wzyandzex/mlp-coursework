"""pytest 配置：把项目根目录加入 sys.path，保证 `import src.*` 与
`import tests.grad_check` 在任何启动方式（python -m pytest / pytest）下都可用。"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
