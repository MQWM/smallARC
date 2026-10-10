from pathlib import Path
import cv2
import pytest
from arrow_solver.models import Arrow,Board,TurnChannel,Plan,Portal
from arrow_solver.vision import read_image,recognize
from arrow_solver.solver import solve,verify_plan,BoardError
from arrow_solver.executor import ScreenValidator,Executor,Cancelled
from arrow_solver.channels import route
from test_next_level import ClockControl


def scene(count=1):
    def arrow(aid,points,d):
        return Arrow(aid,[(round(x/10),round(y/10)) for x,y in points],d,tuple(points[-1]),pixel_path=points)
    arrows = [arrow(1,[(50,25),(50,35)],'DOWN'),
              arrow(2,[(50,5),(50,15)],'DOWN'),
              arrow(3,[(40,120),(60,120)],'RIGHT')]
    c = TurnChannel(1,count,(50,50),(80,80),'DOWN','RIGHT',[(50,50),(50,80),(80,80)],(40,50,40,40))
    return Board(200,200,(0,0,200,200),(0,0),(10,10),20,20,arrows,line_width=4,turn_channels=[c])


def test_depleted_channel_changes_straight_line_dependencies():
    board = scene()
    plan = solve(board)
    assert plan.order==[1,3,2]
    assert plan.channel_uses=={1:[1],3:[],2:[]}
    verify_plan(board,plan)
    bad = Plan([1,2,3],plan.blockers,board.fingerprint(),plan.channel_uses,plan.counter_thresholds)
    with pytest.raises(BoardError,match='仍被阻挡'):
        verify_plan(board,bad)


def test_blocked_exit_does_not_consume_channel():
    board = scene(2)
    blocker = board.arrows[-1]
    blocker.pixel_path = [(100,80),(120,80)]
    blocker.click_point = (120,80)
    plan = solve(board)
    assert plan.order==[3,1,2]
    assert plan.channel_uses[1]==[1] and plan.channel_uses[2]==[1]
    assert board.turn_channels[0].remaining==2


def test_portal_after_turn_stops_blocker_search():
    board = scene()
    board.arrows = [board.arrows[0],Arrow(3,[(15,8),(17,8)],'RIGHT',(170,80),pixel_path=[(150,80),(170,80)])]
    board.portals = [Portal((14,8),(140,80),(135,75,10,10))]
    assert solve(board).order[0]==1


def test_reverse_entry_exits_opposite_end_and_closed_side_is_blocked():
    board = scene()
    arrow = board.arrows[0]
    arrow.pixel_path = [(110,80),(100,80)]
    arrow.direction = 'LEFT'
    segments,uses,blocked = route(board,arrow,{1:1})
    assert not blocked and uses==[1]
    assert segments[-1][0]==(50,50)
    assert segments[-1][1][1]<50
    arrow.pixel_path = [(20,65),(30,65)]
    arrow.direction = 'RIGHT'
    assert route(board,arrow,{1:1})[2]
    assert route(board,arrow,{1:0})[1:]==([],False)


def test_reverse_u_channel_returns_through_left_top_port():
    board = scene()
    board.turn_channels = [TurnChannel(1,2,(50,50),(80,50),'DOWN','UP',
                              [(50,50),(50,80),(80,80),(80,50)],(40,50,50,40))]
    arrow = board.arrows[0]
    arrow.pixel_path = [(80,20),(80,30)]
    segments,uses,blocked = route(board,arrow,{1:2})
    assert not blocked and uses==[1]
    assert segments[-1][0]==(50,50)
    assert segments[-1][1][1]<50


def test_317_reverse_entry_requires_clear_outgoing_path():
    image = read_image(Path(__file__).parent/'fixtures/recorded/level317.png')
    board = recognize(image).board
    assert not board.issues
    assert len(board.arrows)==55
    assert [c.remaining for c in board.turn_channels]==[1,1]
    plan = solve(board)
    assert len(plan.order)==55
    assert {a:u for a,u in plan.channel_uses.items() if u}=={31:[2],14:[1]}
    assert plan.order.index(2)<plan.order.index(14)
    ScreenValidator(image,board)
    verify_plan(board,plan)
    plan.channel_uses[14]=[]
    with pytest.raises(BoardError,match='计划状态不一致'):
        verify_plan(board,plan)


def test_stop_during_channel_wait_resumes_without_repeating_click(tmp_path):
    image = read_image(Path(__file__).parent/'fixtures/recorded/level316.png')
    board = recognize(image).board
    plan = solve(board)
    clock = ClockControl()
    taps = []
    class Device:
        serial='test-only'
        def tap(self,*point):
            taps.append((point,clock.t))
        def screenshot(self):
            pytest.fail('resume must not capture')
    def report(kind,data):
        if kind=='dispatched':
            clock.cancel.set()
    ex = Executor(Device(),board,plan,image,tmp_path,clock,report,delay=.4,now=lambda:clock.t)
    with pytest.raises(Cancelled):
        ex.run()
    assert len(taps)==1
    clock.cancel.clear()
    resumed = Executor(Device(),board,plan,image,tmp_path,clock,delay=.4,
                       not_before=ex.not_before,now=lambda:clock.t)
    assert resumed.run(start=1)==4
    assert len(taps)==4 and len({p for p,_ in taps})==4
    assert taps[1][1]-taps[0][1]>=2


@pytest.mark.parametrize('scale',[.75,1.,1.25])
def test_sparse_316_recognizes_and_plans_four_arrows(scale,tmp_path):
    image = read_image(Path(__file__).parent/'fixtures/recorded/level316.png')
    image = cv2.resize(image,None,fx=scale,fy=scale)
    board = recognize(image).board
    assert not board.issues
    assert len(board.arrows)==4
    assert sorted(c.remaining for c in board.turn_channels)==[2,2]
    assert {c.exit_direction for c in board.turn_channels}=={'UP','LEFT'}
    plan = solve(board)
    assert len(plan.order)==4
    assert all(len(uses)==1 for uses in plan.channel_uses.values())
    ScreenValidator(image,board)
    board.save(tmp_path/'board.json')
    assert Board.load(tmp_path/'board.json').fingerprint()==board.fingerprint()
    clock = ClockControl()
    taps = []
    class Device:
        serial='test-only'
        def tap(self,*point):
            taps.append((point,clock.t))
        def screenshot(self):
            pytest.fail('turn channels must preserve no mid-level capture')
    ex = Executor(Device(),board,plan,image,tmp_path,clock,delay=.4,now=lambda:clock.t)
    assert ex.run()==4
    assert len(taps)==4
    assert all(taps[i+1][1]-taps[i][1]>=2. for i in range(3))
