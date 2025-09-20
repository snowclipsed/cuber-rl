import re
import random
import math
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import verifiers as vf
from verifiers.types import Messages, State
from datasets import Dataset
import magiccube
import kociemba

@dataclass
class CubeState:
    """Rubik's cube state representation"""
    faces: Dict[str, str] = None
    
    FACE_ORDER = ['U', 'L', 'F', 'R', 'B', 'D']
    FACE_NAMES = {'U': 'TOP', 'L': 'LEFT', 'F': 'FRONT', 'R': 'RIGHT', 'B': 'BACK', 'D': 'BOTTOM'}
    COLOR_MAP = {'W': 'U', 'R': 'R', 'G': 'F', 'Y': 'D', 'O': 'L', 'B': 'B'}
    
    def __post_init__(self):
        if self.faces is None:
            self.faces = {f: c*9 for f, c in zip(self.FACE_ORDER, 'WOGRBY')}
    
    @classmethod
    def from_string(cls, s: str) -> 'CubeState':
        """Parse from string format"""
        faces = {}
        for line in s.strip().split('\n'):
            if m := re.match(r'\w+\((\w)\): (\w{3}/\w{3}/\w{3})', line):
                faces[m.group(1)] = m.group(2).replace('/', '')
        return cls(faces)
    
    def to_string(self) -> str:
        """Convert to display format"""
        return '\n'.join(
            f"{self.FACE_NAMES[f]}({f}): {self.faces[f][:3]}/{self.faces[f][3:6]}/{self.faces[f][6:9]}" 
            for f in self.FACE_ORDER
        )
    
    def to_magiccube(self) -> str:
        """Convert for magiccube library"""
        return ''.join(self.faces[f] for f in self.FACE_ORDER)
    
    def to_kociemba(self) -> str:
        """Convert for kociemba solver"""
        order = ['U', 'R', 'F', 'D', 'L', 'B']
        return ''.join(self.COLOR_MAP[c] for f in order for c in self.faces[f])
    
    def is_solved(self) -> bool:
        """Check if cube is solved"""
        return all(len(set(colors)) == 1 for colors in self.faces.values())

def parse_moves(s: str) -> List[str]:
    """Extract valid moves from string"""
    return re.findall(r"[UDLRFB][2']?", s.upper())

def validate_moves(moves: List[str]) -> bool:
    """Validate Singmaster notation"""
    return all(re.match(r"^[UDLRFB][2']?$", m) for m in moves)

def apply_sequence(state: CubeState, moves: List[str]) -> CubeState:
    """Apply move sequence to cube"""
    cube = magiccube.Cube(3, state.to_magiccube())
    for move in moves:
        cube.rotate(move)
    
    result = CubeState()
    result.faces = {f: cube.get()[i*9:(i+1)*9] for i, f in enumerate(CubeState.FACE_ORDER)}
    return result

class Solver:
    """Cube distance and solution calculator"""
    _cache = {}
    
    def distance(self, state: CubeState, target: CubeState = None) -> int:
        """Get optimal move count to target (or solved)"""
        if target and not target.is_solved():
            return 20
        if state.is_solved():
            return 0
        
        key = state.to_kociemba()
        if key in self._cache:
            return self._cache[key]
        
        try:
            solution = kociemba.solve(key)
            dist = 0 if not solution else len(solution.split())
            self._cache[key] = dist
            return dist
        except:
            return 20

def generate_scramble(difficulty: str) -> CubeState:
    """Generate scrambled cube at specified difficulty"""
    ranges = {'very_easy': {1,3}, 'easy': (4, 8), 'medium': (9, 14), 'hard': (15, 20)}
    n = random.randint(*ranges.get(difficulty, (1, 20)))
    
    moves = []
    last = None
    opposites = {'U': 'D', 'D': 'U', 'L': 'R', 'R': 'L', 'F': 'B', 'B': 'F'}
    
    for _ in range(n):
        faces = [f for f in 'UDLRFB' if f != last]
        if last and opposites.get(last) in faces and len(faces) > 2:
            faces.remove(opposites[last])
        
        face = random.choice(faces)
        moves.append(face + random.choice(['', '2', "'"]))
        last = face
    
    return apply_sequence(CubeState(), moves)

def parse_response(response: str) -> Optional[List[str]]:
    """Extract moves from LLM response"""
    if m := re.search(r'<move>(.*?)</move>', response, re.DOTALL):
        content = m.group(1).strip()
        if content == "":
            return []
        moves = parse_moves(content)
        if moves and validate_moves(moves):
            return moves
    return None

def generate_prompt(state: CubeState, max_moves: int) -> str:
    """Generate task prompt"""
    return f"""You are solving a 3x3 Rubik's cube.

Current state:
{state.to_string()}

Task: Provide up to {max_moves} moves to make progress toward solving this cube.

Notation (Singmaster):
- Each letter represents rotating one face of the cube
- U (Up/top), D (Down/bottom), L (Left), R (Right), F (Front), B (Back)
- Single letter = rotate that face 90° clockwise (when looking at that face)
- Letter + ' = rotate that face 90° counterclockwise (e.g., U' rotates top counterclockwise)
- Letter + 2 = rotate that face 180° (e.g., F2 rotates front 180°)

Rules:
- Put your moves in <move>...</move> tags
- Multiple moves separated by spaces
- Use <move></move> if cube is already solved

Think simply and do not overcomplicate. Be concise and only respond with moves in proper format and no other text."""

# Episode Setup
def prepare_episode(x, difficulties=['easy', 'medium'], max_moves_per_turn=3, max_steps=20):
    """Initialize episode with scrambled cube"""
    solver = Solver()
    difficulty = random.choice(difficulties)
    state = generate_scramble(difficulty)
    
    x["task"] = "rubiks-cube"
    x["info"] = {
        "cube": state.faces,
        "initial_dist": solver.distance(state),
        "max_turns": math.ceil(max_steps / max_moves_per_turn),
        "max_moves": max_moves_per_turn,
        "difficulty": difficulty
    }
    x["prompt"] = [{"role": "user", "content": generate_prompt(state, max_moves_per_turn)}]
    return x

class RubiksCubeEnv(vf.MultiTurnEnv):
    """Multi-turn Rubik's cube environment"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.solver = Solver()
    
    def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        """Check episode termination"""
        cube = CubeState(state['info'].get('cube'))
        return cube.is_solved() or state['turn'] >= state['info'].get('max_turns', 10)
    
    def env_response(self, messages: Messages, state: State, **kwargs) -> Tuple[Messages, State]:
        """Process turn and calculate rewards"""
        if not messages or messages[-1]['role'] != 'assistant':
            return [], state
        
        info = state['info']
        response = messages[-1]['content']
        moves = parse_response(response)
        
        turn_reward = 0
        
        # Format reward
        if moves is not None:
            turn_reward += 0.1
        else:
            state['reward'] = 0
            state['total_reward'] = state.get('total_reward', 0)
            msg = "Invalid format. Use <move>...</move> tags. Format for N moves: <move> Move1 Move2 Move3 ... Move N</move>"
            return [{"role": "user", "content": msg}], state
        
        # Handle no moves
        if not moves:
            state['reward'] = turn_reward
            state['total_reward'] = state.get('total_reward', 0) + turn_reward
            current = CubeState(info['cube'])
            msg = f"No moves executed.\n\nCurrent state:\n{current.to_string()}"
            return [{"role": "user", "content": msg}], state
        
        # Execute moves
        moves = moves[:info['max_moves']]
        current = CubeState(info['cube'])
        initial_dist = self.solver.distance(current)
        new_cube = apply_sequence(current, moves)
        final_dist = self.solver.distance(new_cube)
        
        # Progress reward
        if initial_dist > 0:
            progress = max(0, (initial_dist - final_dist) / initial_dist) # we clamp to zero to avoid negative rewards
            turn_reward += progress
        
        info['cube'] = new_cube.faces
        state['reward'] = turn_reward
        state['total_reward'] = state.get('total_reward', 0) + turn_reward
        
        if new_cube.is_solved():
            # Success reward + efficiency reward
            state['total_reward'] += 1.0 + (1.0 / state['turn'])
            return [{"role": "user", "content": f"Solved! Reward: {state['total_reward']:.2f}"}], state
        
        msg = f"""Moves: {' '.join(moves)} | Reward: {turn_reward:.2f}

    Current state:
    {new_cube.to_string()}

    Provide up to {info['max_moves']} moves using <move>...</move> tags. Only respond with moves. Format for N moves: <move> Move1 Move2 Move3 ... Move N</move>"""
        
        return [{"role": "user", "content": msg}], state

def load_environment(**kwargs) -> vf.Environment:
    """Load Rubik's Cube RL environment"""
    dataset = Dataset.from_dict({'episode': range(1000)})
    dataset = dataset.map(
        lambda x: prepare_episode(
            x, 
            kwargs.get('difficulties', ['easy', 'medium']),
            kwargs.get('max_moves_per_turn', 3),
            kwargs.get('max_episode_steps', 20)
        )
    )
    
    parser = vf.Parser()
    def reward_func(parser, completion, info, state=None, **kwargs):
        """Extract total reward from state"""
        if state is None:
            state = kwargs.get('state', {})
        return state.get('total_reward', 0.0)
    
    rubric = vf.Rubric(parser=parser, funcs=[reward_func])
    
    return RubiksCubeEnv(dataset=dataset, parser=parser, rubric=rubric, **kwargs)