"""边界扫描：确保 deploy/app 生产世界不 import 世界外模块。

用法:
    python deploy/tools/boundary_scan.py [--root deploy/app]

扫描 ``deploy/app`` 下所有 ``.py`` 的 import（ast 解析，不导入生产包），
若出现禁用模块即打印越界点并以非 0 退出。纯标准库实现。
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

# 禁用模块：世界外（research/legacy）实现。命中本身或其子模块即越界。
FORBIDDEN = (
    "night_lamp.server_plugin",
    "night_lamp.detector",
    "night_lamp.adaptive",
    "night_lamp.night_state",
    "night_lamp.evidence",
    "night_lamp.main",
    "night_lamp.camera",
    "server.evidence",
    "server.regression",
)

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "app"


def _resolved_imports(tree: ast.AST, pkg_parts):
    """产出 (行号, 解析后的绝对模块名) 列表。"""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ""
            else:
                # level=1 指当前包；逐级上跳。
                up = node.level - 1
                base_parts = pkg_parts[: len(pkg_parts) - up] if up else pkg_parts
                base = ".".join(base_parts)
                if node.module:
                    base = f"{base}.{node.module}" if base else node.module
            out.append((node.lineno, base))
            for alias in node.names:
                if alias.name != "*":
                    out.append((node.lineno, f"{base}.{alias.name}" if base else alias.name))
    return out


def _is_forbidden(name: str) -> bool:
    return any(name == bad or name.startswith(bad + ".") for bad in FORBIDDEN)


def scan(root: Path):
    """返回 [(文件, 行号, 模块名)] 越界列表。"""
    hits = []
    if not root.is_dir():
        return hits, 0
    files = [p for p in sorted(root.rglob("*.py")) if "__pycache__" not in p.parts]
    for fp in files:
        rel = fp.relative_to(root)
        pkg_parts = rel.parts[:-1]  # 文件所在包的组件
        try:
            tree = ast.parse(fp.read_text(encoding="utf-8"), filename=str(fp))
        except SyntaxError as exc:
            print(f"[warn] parse failed {fp}: {exc}", file=sys.stderr)
            continue
        for lineno, name in _resolved_imports(tree, pkg_parts):
            if _is_forbidden(name):
                hits.append((fp, lineno, name))
    return hits, len(files)


def main(argv=None):
    ap = argparse.ArgumentParser(description="deploy/app 边界扫描")
    ap.add_argument("--root", default=str(DEFAULT_ROOT),
                    help="生产世界根目录（默认 deploy/app）")
    args = ap.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"[warn] production world dir missing, nothing to scan: {root}")
        return 0

    hits, nfiles = scan(root)
    if hits:
        print(f"boundary FAIL: {len(hits)} forbidden import(s) in {nfiles} file(s)")
        for fp, lineno, name in hits:
            print(f"  {fp}:{lineno}: {name}")
        return 1

    print(f"boundary OK: {nfiles} file(s), no forbidden module import")
    return 0


if __name__ == "__main__":
    sys.exit(main())
