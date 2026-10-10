from __future__ import annotations

import time

from .executor import Executor,ExecutionError
from .next_level import NextLevel
from .renderer import export_run
from .solver import solve
from .vision import recognize
from .orbit import capture_pair


class ContinuousPlay:
    """One frozen plan per level, with an optional second frame before execution."""
    def __init__(self,device,control,run_root,report,*,delay=.4,flight_wait=5.,
                 next_report=None,now=time.monotonic,orbit_speed=118.,orbit_window=3.,orbit_clockwise=True):
        self.device,self.control,self.run_root,self.report = device,control,run_root,report
        self.delay,self.flight_wait,self.now = delay,flight_wait,now
        self.next_report = next_report or report
        self.orbit_options = dict(orbit_speed=orbit_speed,orbit_window=orbit_window,orbit_clockwise=orbit_clockwise)

    def run(self):
        while True:
            self.control.check()
            result = NextLevel(self.device,self.control,self.next_report,now=self.now).run()
            sample = capture_pair(self.device,self.control,first_image=result.image,
                                  first_at=result.captured_at,now=self.now,report=self.report)
            self.control.check()
            self.report('auto_level_source',{'image':sample.image,'captured_at':sample.captured_at,
                                             'obstacle':sample.obstacle,'warning':sample.warning})
            self.control.check()
            board = recognize(sample.image).board
            options = self.orbit_options.copy()
            if sample.obstacle:
                board.orbit_obstacle = sample.obstacle
                if sample.warning:
                    board.issues.append('小豆豆测速失败：'+sample.warning)
                else:
                    options.update(orbit_speed=sample.obstacle.measured_speed,
                                   orbit_clockwise=sample.obstacle.measured_clockwise)
            self.control.check()
            self.report('auto_level_recognized',{'board':board})
            try:
                if board.issues:
                    raise ExecutionError('新关卡识别存在问题，连续执行已停止：'+'；'.join(board.issues))
                plan = solve(board)
            except (ValueError,ExecutionError):
                export_run(self.run_root,sample.image,board,None)
                raise
            self.control.check()
            run_dir = export_run(self.run_root,sample.image,board,plan)
            self.report('run_directory',{'path':str(run_dir)})
            self.control.check()
            executor = Executor(self.device,board,plan,sample.image,run_dir,self.control,
                                self.report,self.delay,timer_duration=board.timer_seconds,
                                flight_wait=self.flight_wait,now=self.now,captured_at=sample.captured_at,**options)
            self.report('auto_level_plan',{'plan':plan})
            self.control.check()
            executor.run()
            self.next_report = lambda kind,data,executor=executor: executor.log(kind,**data)
