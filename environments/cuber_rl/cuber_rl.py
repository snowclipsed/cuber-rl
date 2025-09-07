import re
import random
import math
from typing import Tuple, List, Optional, Dict
import numpy as np
import verifiers as vf
import magiccube
import kociemba

# State Management
class CubeState:
    FACE_ORDER = ['U', 'L', 'F', 'R', 'B', 'D']
    FACE_NAMES = {'U': 'TOP', 'L': 'LEFT', 'F': 'FRONT', 'R': 'RIGHT', 'B': 'BACK', 'D': 'BOTTOM'}
    COLOR_MAP = {0: 'W', 1: 'O', 2: 'G', 3: 'R', 4: 'B', 5: 'Y'}
    
    def __init__(self, state_string: str = None):
        if state_string:
            self.faces = {}
            for line in state_string.strip().split('\n'):
                match = re.match(r'(\w+)\((\w)\): (\w{3}/\w{3}/\w{3})', line)
                if match:
                    face = match.group(2)
                    colors = match.group(3).replace('/', '')
                    self.faces[face] = colors
        else:
            self.faces = {f: self.COLOR_MAP[i] * 9 for i, f in enumerate(self.FACE_ORDER)}
    
    def to_string(self) -> str:
        return '\n'.join(f"{self.FACE_NAMES[f]}({f}): {self.faces[f][:3]}/{self.faces[f][3:6]}/{self.faces[f][6:9]}" 
                        for f in self.FACE_ORDER)
    
    def to_kociemba(self) -> str:
        kociemba_order = ['U', 'R', 'F', 'D', 'L', 'B']
        kociemba_colors = {'W': 'U', 'R': 'R', 'G': 'F', 'Y': 'D', 'O': 'L', 'B': 'B'}
        return ''.join(kociemba_colors[self.faces[f][i]] for f in kociemba_order for i in range(9))
    
    def from_magiccube(self, cube: magiccube.Cube):
        for i, face_key in enumerate(self.FACE_ORDER):
            face_idx = ['U', 'L', 'F', 'R', 'B', 'D'].index(face_key)
            face_colors = cube.get_face(face_key)
            self.faces[face_key] = ''.join(self.COLOR_MAP[face_colors[r, c]] 
                                          for r in range(3) for c in range(3))
    
    def is_solved(self) -> bool:
        return all(len(set(colors)) == 1 for colors in self.faces.values())
    
    def __eq__(self, other) -> bool:
        return self.faces == other.faces if isinstance(other, CubeState) else False

# Move Operations
def parse_moves(move_string: str) -> List[str]:
    moves = re.findall(r"[UDLRFB][2']?", move_string.upper())
    return [m.replace("'", "_prime") for m in moves]

def validate_moves(moves: List[str]) -> bool:
    valid = set(['U', 'D', 'L', 'R', 'F', 'B'] + 
                [f + s for f in 'UDLRFB' for s in ['2', '_prime']])
    return all(m in valid for m in moves)

def apply_moves(state: CubeState, moves: List[str]) -> CubeState:
    cube = magiccube.Cube(3)
    # Set cube to match state
    for face_key in state.FACE_ORDER:
        face_idx = ['U', 'L', 'F', 'R', 'B', 'D'].index(face_key)
        face_array = np.array([[state.faces[face_key][i*3+j] for j in range(3)] 
                               for i in range(3)])
        color_to_num = {v: k for k, v in CubeState.COLOR_MAP.items()}
        face_nums = np.vectorize(color_to_num.get)(face_array)
        for r in range(3):
            for c in range(3):
                cube.cube[face_idx, r, c] = face_nums[r, c]
    
    for move in moves:
        cube.rotate(move)
    
    new_state = CubeState()
    new_state.from_magiccube(cube)
    return new_state

# Solver/Distance Module  
class CubeSolver:
    def get_distance(self, state: CubeState, target: CubeState = None) -> int:
        if target and target != CubeState():
            return 20  # Max for non-solved targets
        try:
            solution = kociemba.solve(state.to_kociemba())
            return len(solution.split()) if solution else 0
        except:
            return 20
    
    def get_solution(self, state: CubeState) -> str:
        try:
            return kociemba.solve(state.to_kociemba())
        except:
            return ""

# Scramble Generator
def generate_scramble(difficulty: str) -> CubeState:
    ranges = {'easy': (1, 6), 'medium': (7, 14), 'hard': (15, 20)}
    min_moves, max_moves = ranges.get(difficulty, (1, 20))
    num_moves = random.randint(min_moves, max_moves)
    
    moves = []
    last_face = None
    for _ in range(num_moves):
        face = random.choice([f for f in 'UDLRFB' if f != last_face])
        modifier = random.choice(['', '2', "'"])
        moves.append(face + modifier)
        last_face = face
    
    state = CubeState()
    return apply_moves(state, parse_moves(' '.join(moves)))

# LLM Response Parser
def parse_llm_response(response: str) -> Tuple[Optional[List[str]], Optional[CubeState]]:
    move_match = re.search(r'<move>(.*?)</move>', response, re.DOTALL)
    state_match = re.search(r'<state>(.*?)</state>', response, re.DOTALL)
    
    moves = None
    if move_match:
        moves = parse_moves(move_match.group(1))
        if not validate_moves(moves):
            moves = None
    
    predicted_state = None
    if state_match:
        try:
            predicted_state = CubeState(state_match.group(1))
        except:
            predicted_state = None
    
    return moves, predicted_state

# Reward Calculator
def calculate_rewards(initial: CubeState, final: CubeState, moves: List[str], 
                     predicted: Optional[CubeState], solver: CubeSolver) -> Dict:
    progress = (solver.get_distance(initial) - solver.get_distance(final)) / len(moves) if moves else 0
    prediction = -solver.get_distance(predicted, final) / 20 if predicted else 0
    return {'progress': progress, 'prediction': prediction, 'total': 0.7 * progress + 0.3 * prediction}

# Environment Wrapper
class RubiksCubeEnv(vf.Environment):
    def __init__(self, difficulties: List[str] = ['easy', 'medium'], 
                 max_moves_per_turn: int = 3, max_episode_steps: int = 20):
        super().__init__()
        self.difficulties = difficulties
        self.max_moves = max_moves_per_turn
        self.max_episode_steps = max_episode_steps
        self.solver = CubeSolver()
        self.reset()
    
    def reset(self) -> str:
        self.state = generate_scramble(random.choice(self.difficulties))
        self.initial_distance = self.solver.get_distance(self.state)
        self.max_turns = math.ceil(self.max_episode_steps / self.max_moves)
        self.turn = 0
        self.done = False
        self.total_reward = 0
        return self.get_observation()
    
    def get_observation(self) -> str:
        return f"""Current cube state:
{self.state.to_string()}

Task: Provide the next {self.max_moves} most optimal moves to solve the cube.
- Use standard notation (U, D, L, R, F, B with optional ', 2)
- Put moves in <move>...</move> tags
- Predict final state after moves in <state>...</state> tags
Turn {self.turn + 1}/{self.max_turns}"""
    
    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        moves, predicted = parse_llm_response(action)
        
        if not moves:
            reward = -1.0
            self.done = True
            info = {'error': 'Invalid move format'}
        else:
            if len(moves) > self.max_moves:
                moves = moves[:self.max_moves]
            
            new_state = apply_moves(self.state, moves)
            rewards = calculate_rewards(self.state, new_state, moves, predicted, self.solver, self.max_episode_steps)
            reward = rewards['total']
            
            self.state = new_state
            self.turn += 1
            self.done = self.state.is_solved() or self.turn >= self.max_turns
            info = {'moves': moves, 'rewards': rewards, 'solved': self.state.is_solved()}
        
        self.total_reward += reward
        return self.get_observation(), reward, self.done, info
    
    def render(self) -> str:
        return f"Distance to solved: {self.solver.get_distance(self.state)}\n{self.state.to_string()}"

def load_environment(**kwargs) -> vf.Environment:
    '''Loads a custom Rubik's Cube RL environment.'''
    return RubiksCubeEnv(
        difficulties=kwargs.get('difficulties', ['easy', 'medium']),
        max_moves_per_turn=kwargs.get('max_moves_per_turn', 3),
        max_episode_steps=kwargs.get('max_episode_steps', 30)
    )