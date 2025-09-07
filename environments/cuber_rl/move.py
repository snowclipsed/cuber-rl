import re
from magiccube import Cube
from state import CubeState

def parse_moves(move_string):
    """Parse 'U R' D2 F' into ['U', 'R'', 'D2', 'F']"""
    return re.findall(r"[UDLRFB][2']?", move_string.replace("'", "'"))

def validate_moves(moves):
    """Check if all moves are valid Singmaster notation"""
    valid_pattern = re.compile(r"^[UDLRFB][2']?$")
    return all(valid_pattern.match(m) for m in moves)

def apply_moves(state, moves):
    """Apply move sequence to state using magiccube"""
    if not validate_moves(moves):
        raise ValueError(f"Invalid moves: {moves}")
    
    # Convert to magiccube
    cube = Cube(3)
    cube._state = state.to_magiccube()
    
    # Apply each move
    for move in moves:
        face = move[0]
        if len(move) == 2:
            if move[1] == '2':
                cube.rotate(face, 2)
            elif move[1] == "'":
                cube.rotate(face, -1)
        else:
            cube.rotate(face, 1)
    
    return CubeState.from_magiccube(cube._state)

def normalize_moves(moves):
    """Remove redundant moves like R R' or R R R R"""
    if not moves:
        return []
    
    result = []
    for move in moves:
        if result and result[-1][0] == move[0]:
            # Same face as previous move
            prev = result.pop()
            combined = _combine_moves(prev, move)
            if combined:
                result.append(combined)
        else:
            result.append(move)
    return result

def _combine_moves(m1, m2):
    """Combine two moves on same face"""
    turns = {'': 1, '2': 2, "'": -1}
    total = (turns.get(m1[1:], 1) + turns.get(m2[1:], 1)) % 4
    
    if total == 0:
        return None
    face = m1[0]
    return face + {1: '', 2: '2', 3: "'"}[total]