from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description="小小箭头桌面辅助工具")
    parser.add_argument("--analyze", type=Path, help="离线识别截图并导出，不启动桌面窗口")
    parser.add_argument("--crop", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--runs", type=Path, default=Path.home() / "SmallARC-runs" if getattr(sys, "frozen", False) else Path.cwd() / "runs")
    args = parser.parse_args()
    if args.analyze:
        from .vision import read_image, recognize
        from .solver import solve
        from .renderer import export_run
        image = read_image(args.analyze)
        result = recognize(image, args.crop)
        try:
            plan = solve(result.board)
        except ValueError as exc:
            plan = None
            result.board.issues.append(str(exc))
        path = export_run(args.runs, image, result.board, plan)
        print(f"箭头 {len(result.board.arrows)}，网格 {result.board.cols}×{result.board.rows}，输出：{path}")
        for issue in result.board.issues:
            print("需检查：", issue)
        if plan:
            print("顺序：", plan.order)
        if result.board.issues or plan is None:
            raise SystemExit(2)
        return
    from PySide6.QtWidgets import QApplication
    from .ui.main_window import MainWindow
    application = QApplication(sys.argv[:1])
    application.setStyle("Fusion")
    window = MainWindow(args.runs)
    window.show()
    sys.exit(application.exec())


if __name__ == "__main__":
    main()
