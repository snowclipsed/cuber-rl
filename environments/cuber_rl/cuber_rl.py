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
        """Convert to unfolded net display format"""
        u = self.faces['U']
        l = self.faces['L']
        f = self.faces['F']
        r = self.faces['R']
        b = self.faces['B']
        d = self.faces['D']
        
        return f"""
        {u[0]} {u[1]} {u[2]}
        {u[3]} {u[4]} {u[5]}
        {u[6]} {u[7]} {u[8]}

{l[0]} {l[1]} {l[2]}   {f[0]} {f[1]} {f[2]}   {r[0]} {r[1]} {r[2]}   {b[0]} {b[1]} {b[2]}
{l[3]} {l[4]} {l[5]}   {f[3]} {f[4]} {f[5]}   {r[3]} {r[4]} {r[5]}   {b[3]} {b[4]} {b[5]}
{l[6]} {l[7]} {l[8]}   {f[6]} {f[7]} {f[8]}   {r[6]} {r[7]} {r[8]}   {b[6]} {b[7]} {b[8]}

        {d[0]} {d[1]} {d[2]}
        {d[3]} {d[4]} {d[5]}
        {d[6]} {d[7]} {d[8]}"""

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
    ranges = {'very_easy': (1,3), 'easy': (4, 8), 'medium': (9, 14), 'hard': (15, 20)}
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

def generate_prompt(state: CubeState, max_moves: int, distance: int = None) -> str:
    """Generate task prompt"""
    dist_info = f" ({distance} moves from solved)" if distance is not None else ""
    
    return f"""You are solving a 3x3 Rubik's cube.

Solved state (goal):
        W W W
        W W W
        W W W

O O O   G G G   R R R   B B B
O O O   G G G   R R R   B B B
O O O   G G G   R R R   B B B

        Y Y Y
        Y Y Y
        Y Y Y

Current state{dist_info}:
{state.to_string()}

Task: Provide up to {max_moves} moves to make progress toward solving this cube.

Notation (Singmaster):
- U (Up/top), D (Down/bottom), L (Left), R (Right), F (Front), B (Back)
- Single letter = rotate that face 90° clockwise
- Letter + ' = rotate 90° counterclockwise (e.g., U')
- Letter + 2 = rotate 180° (e.g., F2)

Rules:
- Put your moves in <move>...</move> tags
- Multiple moves separated by spaces
- Use <move></move> if cube is already solved

Think simply and do not overcomplicate. Be concise and only respond with {max_moves} moves in proper format and no other text."""

# Episode Setup
def prepare_episode(x, difficulties=['easy', 'medium'], max_moves_per_turn=3, max_steps=20):
    """Initialize episode with scrambled cube"""
    solver = Solver()
    difficulty = random.choice(difficulties)
    state = generate_scramble(difficulty)
    initial_dist = solver.distance(state)
    
    x["task"] = "rubiks-cube"
    x["info"] = {
        "cube": state.faces,
        "initial_dist": initial_dist,
        "max_turns": math.ceil(max_steps / max_moves_per_turn),
        "max_moves": max_moves_per_turn,
        "difficulty": difficulty
    }
    x["prompt"] = [{"role": "user", "content": generate_prompt(state, max_moves_per_turn, initial_dist)}]
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
        """Process turn and calculate rewards with positive-only structure"""
        if not messages or messages[-1]['role'] != 'assistant':
            return [], state

        info = state['info']
        response = messages[-1]['content']
        moves = parse_moves(response)

        turn_reward = 0.0

        # Format reward
        if moves is not None:
            turn_reward += 0.1
        else:
            state['reward'] = 0.0
            state['total_reward'] = state.get('total_reward', 0.0)
            msg = "Invalid format. Use <move>...</move> tags. Format for N moves: <move> Move1 Move2 Move3 ... Move N</move>"
            return [{"role": "user", "content": msg}], state

        # Track initial distance
        if 'initial_distance' not in info:
            current = CubeState(info['cube'])
            info['initial_distance'] = self.solver.distance(current)

        d = info['initial_distance']

        # No-op handling
        if not moves:
            state['reward'] = turn_reward
            state['total_reward'] = state.get('total_reward', 0.0) + turn_reward
            current = CubeState(info['cube'])
            msg = f"No moves executed.\n\nCurrent state:\n{current.to_string()}"
            return [{"role": "user", "content": msg}], state

        # Execute moves
        moves = moves[:info['max_moves']]
        current = CubeState(info['cube'])
        new_cube = apply_sequence(current, moves)
        final_distance = self.solver.distance(new_cube)

        turns_used = state.get('turn', 1)

        # Calculate rewards based on solved status
        if new_cube.is_solved():
            solve_reward = 1.0
            efficiency_ratio = min(1.0, d / turns_used)
            efficiency_reward = efficiency_ratio
            distance_reward = 0.0
            
            turn_reward += solve_reward + efficiency_reward + distance_reward
            info['cube'] = new_cube.faces
            state['reward'] = turn_reward
            state['total_reward'] = state.get('total_reward', 0.0) + turn_reward
            return [{"role": "user", "content": f"Solved! Reward: {state['total_reward']:.2f}"}], state
        else:
            solve_reward = 0.0
            efficiency_reward = 0.0
            net_reduction = max(0, d - final_distance)
            progress_ratio = net_reduction / d if d > 0 else 0.0
            distance_reward = progress_ratio
            
            turn_reward += distance_reward

        # Update state
        info['cube'] = new_cube.faces
        state['reward'] = turn_reward
        state['total_reward'] = state.get('total_reward', 0.0) + turn_reward

        msg = f"""Moves: {' '.join(moves)} | Reward: {turn_reward:.4f} | Distance from solved: {final_distance}

        Current state:
        {new_cube.to_string()}

        Provide up to {info['max_moves']} moves using <move>...</move> tags. Only respond with moves. Format for N moves: <move> Move1 Move2 Move3 ... Move N</move>"""

        return [{"role": "user", "content": msg}], state

def load_environment(difficulties=['easy', 'medium'], max_moves_per_turn=3, max_episode_steps=20) -> vf.Environment:
    """Load Rubik's Cube RL environment"""
    dataset = Dataset.from_dict({'episode': range(1000)})
    dataset = dataset.map(
        lambda x: prepare_episode(
            x, 
            difficulties,
            max_moves_per_turn,
            max_episode_steps
        )
    )
    
    parser = vf.Parser()
    def reward_func(parser, completion, info, state=None, **kwargs):
        """Extract total reward from state"""
        if state is None:
            state = kwargs.get('state', {})
        return state.get('total_reward', 0.0)
    
    rubric = vf.Rubric(parser=parser, funcs=[reward_func])
    
    return RubiksCubeEnv(dataset=dataset, parser=parser, rubric=rubric)