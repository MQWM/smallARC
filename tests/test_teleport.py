from copy import deepcopy
from pathlib import Path
import cv2
import pytest

from arrow_solver.models import Board,Arrow,TeleportGate,Portal,CounterBlock,TurnChannel
from arrow_solver.teleport import find_gates
from arrow_solver.vision import read_image,recognize,suggest_crop
from arrow_solver.solver import solve,verify_plan,BoardError
from arrow_solver.channels import route,move
from arrow_solver.executor import ScreenValidator,Executor,ExecutionError
from arrow_solver.renderer import overlay,export_run
from test_next_level import ClockControl

FIXTURES = Path(__file__).parent/'fixtures/recorded'


def scene():
    a = Arrow(1,[(2,5),(3,5)],'RIGHT',(30,50),pixel_path=[(20,50),(30,50)])
    return Board(300,200,(0,0,300,200),(0,0),(10,10),30,20,[a],line_width=4,
                 teleport_gates=[TeleportGate(1,(60,50),(55,40,10,20),2),
                                 TeleportGate(2,(150,100),(145,90,10,20),1)])


@pytest.mark.parametrize('scale',[.75,1.,1.25])
def test_409_complete_recognition_plan_and_preflight(scale,tmp_path):
    im = read_image(FIXTURES/'level409.png')
    im = cv2.resize(im,None,fx=scale,fy=scale)
    board = recognize(im).board
    assert len(board.arrows)==43 and not board.issues
    assert len(board.teleport_gates)==2
    assert not board.portals and not board.turn_channels
    assert [g.partner_id for g in board.teleport_gates]==[2,1]
    plan = solve(board)
    verify_plan(board,plan)
    ScreenValidator(im,board)
    board.save(tmp_path/'board.json')
    assert Board.load(tmp_path/'board.json').fingerprint()==board.fingerprint()
    clock = ClockControl()
    taps = []
    class Device:
        serial = 'fake'
        def tap(self,*point):taps.append(point)
        def screenshot(self):pytest.fail('no screenshots during teleport execution')
    assert Executor(Device(),board,plan,im,tmp_path,clock,delay=0,now=lambda:clock.t).run()==43
    assert len(taps)==43
    if scale==1:
        # These three upward arrows enter the upper-right gate, leave lower-left.
        assert plan.blockers[19]==[] and plan.blockers[23]==[19]
        assert plan.order.index(19)<plan.order.index(23)<plan.order.index(41)
        preview = overlay(im,board,plan,plan.order.index(19))
        assert preview.shape==im.shape
        import json
        run = export_run(tmp_path/'records',im,board,plan)
        assert len(json.loads((run/'solution.json').read_text())['teleport_gates'])==2
        missing = deepcopy(board)
        missing.teleport_gates.clear()
        with pytest.raises(ExecutionError,match='未解释'):
            ScreenValidator(im,missing)


def test_entry_and_exit_blockers_count_but_teleport_gap_does_not():
    board = scene()
    def body(aid,x,y):
        return Arrow(aid,[(x//10,y//10),(x//10,(y+10)//10)],'DOWN',(x,y+10),pixel_path=[(x,y),(x,y+10)])
    board.arrows += [body(2,45,45),body(3,180,95),body(4,100,65)]
    blocked,blockers,_,_ = move(board,board.arrows[0],{1,2,3,4},{})
    assert not blocked and set(blockers)=={2,3}
    segments,_,blocked = route(board,board.arrows[0],{})
    assert not blocked and len(segments)==2
    assert segments[0][1]==(55.,50.) and segments[1][0]==(150,100)
    plan = solve(board)
    assert plan.order.index(2)<plan.order.index(1) and plan.order.index(3)<plan.order.index(1)
    verify_plan(board,plan)
    a = board.arrows[0]
    a.pixel_path=[(185,100),(170,100)];a.direction='LEFT'
    assert route(board,a,{})[0][1][0]==(60,50)
    assert all(g.partner_id for g in board.teleport_gates)


def test_absorber_and_counter_after_teleport_are_respected():
    board = scene()
    board.portals=[Portal((20,10),(200,100),(195,95,10,10))]
    board.counter_blocks=[CounterBlock((18,10),2,(175,95,10,10))]
    assert route(board,board.arrows[0],{})[0][-1][1]==(200.,100.)
    assert move(board,board.arrows[0],{1},{})[-1]==2


def test_unpaired_gate_and_teleport_loop_are_rejected():
    board = scene()
    board.teleport_gates[1].partner_id=None
    with pytest.raises(BoardError,match='配对'):
        solve(board)
    board = scene()
    board.teleport_gates[1].center=(10,50)
    board.teleport_gates[1].visual_rect=(5,40,10,20)
    assert route(board,board.arrows[0],{})[2]
    with pytest.raises(BoardError):solve(board)


def test_gate_then_turn_channel_preserves_dynamic_routing():
    board = scene()
    board.turn_channels=[TurnChannel(1,1,(180,100),(200,120),'RIGHT','DOWN',
                                    [(180,100),(200,100),(200,120)],(180,90,30,30))]
    segments,uses,blocked = route(board,board.arrows[0],{1:1})
    assert not blocked and uses==[1]
    assert segments[-1][1][0]==200 and segments[-1][1][1]>120
    assert route(board,board.arrows[0],{1:0})[1:]==([],False)


def test_reentering_gate_from_another_direction_is_not_a_loop():
    board = scene()
    board.teleport_gates=[TeleportGate(1,(90,50),(85,40,10,20),2),
                          TeleportGate(2,(60,100),(55,90,10,20),1)]
    board.turn_channels=[TurnChannel(1,1,(80,100),(90,95),'RIGHT','UP',
                                    [(80,100),(90,100),(90,95)],(80,95,20,20))]
    segments,uses,blocked = route(board,board.arrows[0],{1:1})
    assert not blocked and uses==[1]
    assert segments[-1][0]==(60,100)
    assert segments[-1][1][0]==60 and segments[-1][1][1]<0


@pytest.mark.parametrize('name',['level316.png','level317.png','level203.png','level205.png','level342.png','level262-victory.png'])
def test_existing_channels_absorbers_and_electric_are_not_teleport_gates(name):
    im = read_image(FIXTURES/name)
    assert find_gates(im,suggest_crop(im))==[]
