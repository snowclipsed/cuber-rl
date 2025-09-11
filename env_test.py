import random
from environments.cuber_rl.cuber_rl import *

# State Tests
def test_cube_state_creation():
    """Test state initialization and parsing"""
    # Default solved state
    cube = CubeState()
    assert cube.is_solved()
    assert cube.faces['U'] == 'W' * 9
    
    # Parse from string
    state_str = "TOP(U): WRB/OWG/YWW\nLEFT(L): BBG/BBR/GGR\nFRONT(F): OOY/OOY/ROO\nRIGHT(R): GYG/RRR/YYO\nBACK(B): RRW/GGW/BWB\nBOTTOM(D): YOB/WWG/WYR"
    cube2 = CubeState.from_string(state_str)
    assert cube2.faces['U'] == 'WRBOWGYWW'
    assert not cube2.is_solved()
    
    # Round trip
    assert CubeState.from_string(cube.to_string()) == cube

def test_state_conversions():
    """Test format conversions"""
    cube = CubeState()
    
    # Magiccube format (54 chars)
    mc = cube.to_magiccube()
    assert len(mc) == 54
    assert mc == 'W'*9 + 'O'*9 + 'G'*9 + 'R'*9 + 'B'*9 + 'Y'*9
    
    # Kociemba format (different face order)
    kc = cube.to_kociemba()
    assert len(kc) == 54
    assert kc[0:9] == 'U'*9  # White -> U

# Move Tests
def test_move_parsing():
    """Test move string parsing"""
    assert parse_moves("U R' F2 D") == ['U', "R'", 'F2', 'D']
    assert parse_moves("urfd2b'") == ['U', 'R', 'F', 'D2', "B'"]
    assert parse_moves("XYZ") == []
    assert validate_moves(['U', "R'", 'F2'])
    assert not validate_moves(['U3', 'X'])

def test_move_application():
    """Test move mechanics"""
    cube = CubeState()
    
    # Single move changes state
    after_u = apply_sequence(cube, ['U'])
    assert after_u != cube
    assert not after_u.is_solved()
    
    # Inverse cancels
    identity = apply_sequence(cube, ['U', "U'"])
    assert identity == cube
    
    # Known pattern (sexy move 6x = identity)
    sexy6 = apply_sequence(cube, ["R", "U", "R'", "U'"] * 6)
    assert sexy6 == cube

# Solver Tests
def test_solver_distance():
    """Test distance calculations"""
    solver = Solver()
    solved = CubeState()
    
    assert solver.distance(solved) == 0
    
    one_move = apply_sequence(solved, ['U'])
    assert solver.distance(one_move) == 1
    
    scrambled = generate_scramble('hard')
    dist = solver.distance(scrambled)
    assert 0 < dist <= 20

def test_solver_cache():
    """Test solver caching"""
    solver = Solver()
    cube = generate_scramble('easy')
    
    # First call
    dist1 = solver.distance(cube)
    # Cached call (check cache hit)
    assert cube.to_kociemba() in solver._cache
    dist2 = solver.distance(cube)
    assert dist1 == dist2

def test_solver_solution():
    """Test solution generation"""
    solver = Solver()
    
    # One move scramble
    cube = apply_sequence(CubeState(), ['U'])
    solution = solver.solve(cube)
    solved = apply_sequence(cube, parse_moves(solution))
    assert solved.is_solved()

# Scramble Tests
def test_scramble_difficulty():
    """Test scramble generation respects difficulty"""
    solver = Solver()
    
    for _ in range(5):
        easy = generate_scramble('easy')
        dist = solver.distance(easy)
        assert 1 <= dist <= 6
        
        hard = generate_scramble('hard')
        dist = solver.distance(hard)
        assert 7 <= dist <= 20

def test_scramble_quality():
    """Test scrambles don't have obvious issues"""
    # Check moves don't immediately cancel
    random.seed(42)
    cube = generate_scramble('medium')
    assert not cube.is_solved()

# LLM Parsing Tests
def test_parse_response():
    """Test LLM response parsing"""
    # Valid response
    resp = "I'll solve this. <move>U R' F2</move> This should help. <state>TOP(U): WWW/WWW/WWW\nLEFT(L): OOO/OOO/OOO\nFRONT(F): GGG/GGG/GGG\nRIGHT(R): RRR/RRR/RRR\nBACK(B): BBB/BBB/BBB\nBOTTOM(D): YYY/YYY/YYY</state>"
    moves, state = parse_response(resp)
    assert moves == ['U', "R'", 'F2']
    assert state.is_solved()
    
    # Empty moves
    resp2 = "<move></move> <state>TOP(U): WWW/WWW/WWW\nLEFT(L): OOO/OOO/OOO\nFRONT(F): GGG/GGG/GGG\nRIGHT(R): RRR/RRR/RRR\nBACK(B): BBB/BBB/BBB\nBOTTOM(D): YYY/YYY/YYY</state>"
    moves2, state2 = parse_response(resp2)
    assert moves2 == []
    assert state2.is_solved()
    
    # Invalid moves
    resp3 = "<move>X Y Z</move>"
    moves3, _ = parse_response(resp3)
    assert moves3 is None
    
    # Missing tags
    resp4 = "Just some text"
    moves4, state4 = parse_response(resp4)
    assert moves4 is None
    assert state4 is None

# Reward Tests
def test_path_reward():
    """Test optimal pathing reward"""
    rc = RewardCalculator()
    
    # Good progress
    assert rc.path_reward(10, 5, 5) == 1.0
    assert rc.path_reward(10, 8, 2) == 1.0
    
    # No progress
    assert rc.path_reward(10, 10, 3) == 0.0
    
    # Regression
    assert rc.path_reward(5, 10, 5) == -1.0
    
    # No moves
    assert rc.path_reward(10, 10, 0) == 0.0

def test_model_reward():
    """Test mental model reward"""
    rc = RewardCalculator()
    cube1 = CubeState()
    cube2 = apply_sequence(cube1, ['U'])
    
    assert rc.model_reward(cube1, cube1) == 1.0
    assert rc.model_reward(cube1, cube2) == -0.5
    assert rc.model_reward(None, cube1) == -0.5

def test_completion_bonus():
    """Test solving bonus calculation"""
    rc = RewardCalculator()
    
    # Early solve gets higher bonus
    assert rc.completion_bonus(2, 10) > rc.completion_bonus(8, 10)
    assert rc.completion_bonus(0, 10) == 2.0
    assert rc.completion_bonus(10, 10) == 0.0

# Environment Tests
def test_episode_preparation():
    """Test episode initialization"""
    x = {'episode': 0}
    prepared = prepare_episode(x, ['easy'], 3, 18)
    
    assert x['task'] == 'rubiks-cube'
    assert 'cube' in x['info']
    assert x['info']['max_moves'] == 3
    assert x['info']['max_turns'] == 6
    assert x['info']['difficulty'] == 'easy'
    assert len(x['prompt']) == 1
    assert 'Current state:' in x['prompt'][0]['content']


# Main Test Runner
def main():
    """Run all tests and report results"""
    tests = [
        test_cube_state_creation,
        test_state_conversions,
        test_move_parsing,
        test_move_application,
        test_solver_distance,
        test_solver_cache,
        test_solver_solution,
        test_scramble_difficulty,
        test_scramble_quality,
        test_parse_response,
        test_path_reward,
        test_model_reward,
        test_completion_bonus,
        test_episode_preparation
    ]
    
    passed = 0
    failed = []
    
    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__}: {e}")
            failed.append(test.__name__)
        except Exception as e:
            print(f"✗ {test.__name__}: {type(e).__name__}: {e}")
            failed.append(test.__name__)
    
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(tests)} passed")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    return len(failed) == 0

if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)