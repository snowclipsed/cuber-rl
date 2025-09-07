
import kociemba
import numpy as np

class CubeSolver:
    def __init__(self):
        # Kociemba uses color mapping: U=0,R=1,F=2,D=3,L=4,B=5
        # Our mapping: W=0,R=1,B=2,O=3,G=4,Y=5
        # Standard cube: U=W,L=O,F=G,R=R,B=B,D=Y
        self.color_map = {0: 'U', 1: 'R', 2: 'B', 3: 'L', 4: 'F', 5: 'D'}
        
    def _to_kociemba_string(self, state):
        """Convert our state to 54-char kociemba format"""
        # Kociemba order: URFDLB, each face in reading order
        koci_order = [0, 3, 2, 1, 4, 5]  # Our indices to kociemba order
        
        chars = []
        for face_idx in koci_order:
            face = state.state[face_idx] if hasattr(state, 'state') else state[face_idx]
            for row in face:
                for cell in row:
                    chars.append(self.color_map[cell])
        return ''.join(chars)
    
    def get_distance(self, state, target_state=None):
        """Get optimal move count between states"""
        if target_state is None:
            # Distance to solved
            koci_string = self._to_kociemba_string(state)
            if koci_string == 'U' * 9 + 'R' * 9 + 'F' * 9 + 'D' * 9 + 'L' * 9 + 'B' * 9:
                return 0
            solution = kociemba.solve(koci_string)
            return len(solution.split()) if solution else 0
        else:
            # Distance between two states - solve composition
            return self._distance_between(state, target_state)
    
    def _distance_between(self, state1, state2):
        """Calculate distance between arbitrary states via inverse composition"""
        # Convert state1 to solved, then solved to state2
        s1_string = self._to_kociemba_string(state1)
        s2_string = self._to_kociemba_string(state2)
        
        # Get moves to solve state1
        s1_to_solved = kociemba.solve(s1_string)
        if not s1_to_solved:
            s1_to_solved = []
        else:
            s1_to_solved = s1_to_solved.split()
        
        # Apply inverse to state2 to get composite problem
        temp_state = state2
        for move in reversed(s1_to_solved):
            inv_move = move[0] + ("'" if len(move) == 1 else "" if move[1] == "'" else move[1])
            temp_state = apply_moves(temp_state, [inv_move])
        
        # Solve the composite
        composite_string = self._to_kociemba_string(temp_state)
        solution = kociemba.solve(composite_string)
        return len(solution.split()) if solution else 0
    
    def get_solution(self, state, max_length=20):
        """Get optimal move sequence to solve"""
        koci_string = self._to_kociemba_string(state)
        solution = kociemba.solve(koci_string)
        
        if not solution:
            return []
        
        moves = solution.split()
        if len(moves) > max_length:
            # Use suboptimal but shorter solution
            solution = kociemba.solve_with_timeout(koci_string, max_length)
            moves = solution.split() if solution else moves[:max_length]
        
        return moves

from state import CubeState
from move import apply_moves
import random

def generate_scramble(difficulty='medium', solver=None):
    """Generate scramble with specific difficulty"""
    ranges = {
        'easy': (1, 6),
        'medium': (7, 14), 
        'hard': (15, 20)
    }
    min_moves, max_moves = ranges.get(difficulty, difficulty)
    
    if not solver:
        solver = CubeSolver()
    
    # Generate random scrambles until we get desired difficulty
    while True:
        num_moves = random.randint(min_moves, max_moves)
        moves = _random_move_sequence(num_moves)
        
        state = CubeState()
        scrambled = apply_moves(state, moves)
        actual_distance = solver.get_distance(scrambled)
        
        if min_moves <= actual_distance <= max_moves:
            return scrambled, moves, actual_distance

def _random_move_sequence(length):
    """Generate non-redundant random move sequence"""
    faces = ['U', 'D', 'L', 'R', 'F', 'B']
    modifiers = ['', '2', "'"]
    opposite = {'U': 'D', 'D': 'U', 'L': 'R', 'R': 'L', 'F': 'B', 'B': 'F'}
    
    moves = []
    prev_face = None
    prev_prev_face = None
    
    while len(moves) < length:
        face = random.choice(faces)
        
        # Avoid same face twice in a row
        if face == prev_face:
            continue
        
        # Avoid opposite faces three times (e.g., U D U is redundant)
        if prev_prev_face and face == prev_prev_face and opposite[face] == prev_face:
            continue
        
        modifier = random.choice(modifiers)
        moves.append(face + modifier)
        prev_prev_face = prev_face
        prev_face = face
    
    return moves

def generate_dataset(difficulties, count_per_difficulty, solver=None):
    """Batch generate scrambles for training"""
    if not solver:
        solver = CubeSolver()
    
    dataset = []
    for difficulty in difficulties:
        for _ in range(count_per_difficulty):
            state, scramble, distance = generate_scramble(difficulty, solver)
            dataset.append({
                'state': state,
                'scramble': scramble,
                'distance': distance,
                'difficulty': difficulty
            })
    
    return dataset