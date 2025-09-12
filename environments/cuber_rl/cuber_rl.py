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

# Core Cube State
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
    
    def __eq__(self, other) -> bool:
        return isinstance(other, CubeState) and self.faces == other.faces

# Move Operations
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

# Solver
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
    
    def solve(self, state: CubeState) -> str:
        """Get solution moves"""
        return "" if state.is_solved() else kociemba.solve(state.to_kociemba())

# Scramble Generation
def generate_scramble(difficulty: str) -> CubeState:
    """Generate scrambled cube at specified difficulty"""
    ranges = {'easy': (1, 6), 'medium': (7, 14), 'hard': (15, 20)}
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

# LLM Interface
def parse_response(response: str) -> Tuple[Optional[List[str]], Optional[CubeState]]:
    """Extract moves and predicted state from LLM response"""
    moves = None
    if m := re.search(r'<move>(.*?)</move>', response, re.DOTALL):
        content = m.group(1).strip()
        if content == "":
            moves = []
        else:
            moves = parse_moves(content)
            if not moves or not validate_moves(moves):
                moves = None
    
    predicted = None
    if m := re.search(r'<state>(.*?)</state>', response, re.DOTALL):
        try:
            predicted = CubeState.from_string(m.group(1))
        except:
            pass
    
    return moves, predicted

def generate_prompt(state: CubeState, max_moves: int) -> str:
    """Generate task prompt"""
    return f"""You are solving a 3x3 Rubik's cube.

State representation: Each face shows its 9 stickers as a 3x3 grid (rows separated by /).
Colors: W=White, R=Red, B=Blue, O=Orange, G=Green, Y=Yellow

Current state:
{state.to_string()}

Task: Provide up to {max_moves} moves to make progress toward solving this cube, and correctly predict the state of the cube after those moves are made.

Rules:
- Please use Singmaster notation: U, D, L, R, F, B (with optional ' for counterclockwise, 2 for double).
- You must put your moves in <move>...</move> tags (use <move></move> if no moves needed).
- You must put the resulting state after your moves in <state>...</state> tags using the same format as provided to you.
- If cube is solved before you reach {max_moves} moves, provide no further moves and just confirm the cube is solved.

Be concise and only respond with the answer, using <move> and <state> tags according to the rules.
"""

# Reward Calculation
class RewardCalculator:
    """Calculate various reward components"""
    
    @staticmethod
    def path_reward(initial_dist: int, final_dist: int, num_moves: int) -> float:
        """Reward for optimal pathing"""
        if num_moves == 0:
            return 0
        return (initial_dist - final_dist) / num_moves
    
    @staticmethod
    def model_reward(predicted: Optional[CubeState], actual: CubeState) -> float:
        """Reward for mental modeling accuracy"""
        if predicted is None:
            return -1.0  # Heavy penalty for not trying
        return 1.0 if predicted == actual else -0.2  # Light penalty for wrong attempt
    
    @staticmethod
    def completion_bonus(turn: int, max_turns: int) -> float:
        """Bonus for solving within turn limit"""
        return 2.0 * (1 - turn / max_turns)

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

# Environment
class RubiksCubeEnv(vf.MultiTurnEnv):
    """Multi-turn Rubik's cube environment"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.solver = Solver()
        self.rewards = RewardCalculator()
    
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
        moves, predicted = parse_response(response)
        
        if moves is None:
            state['reward'] = -0.5
            state['total_reward'] = state.get('total_reward', 0) - 0.5
            msg = "Invalid move format. Provide moves in <move>...</move> tags. Continuing..."
            return [{"role": "user", "content": msg}], state
        
        if not moves:
            state['reward'] = -0.1  # Small penalty for no action
            state['total_reward'] = state.get('total_reward', 0) - 0.1
            current = CubeState(info['cube'])
            msg = f"No moves executed.\n\nCurrent state:\n{current.to_string()}\n\nProvide up to {info['max_moves']} moves in <move>...</move> tags."
            return [{"role": "user", "content": msg}], state
        
        moves = moves[:info['max_moves']]
        current = CubeState(info['cube'])
        initial_dist = self.solver.distance(current)
        new_cube = apply_sequence(current, moves)
        final_dist = self.solver.distance(new_cube)
        
        path_r = self.rewards.path_reward(initial_dist, final_dist, len(moves))
        model_r = self.rewards.model_reward(predicted, new_cube)
        
        state['reward'] = 0.7 * path_r + 0.3 * model_r
        state['total_reward'] = state.get('total_reward', 0) + state['reward']
        info['cube'] = new_cube.faces
        
        if new_cube.is_solved():
            bonus = self.rewards.completion_bonus(state['turn'], info['max_turns'])
            state['total_reward'] += bonus
            return [{"role": "user", "content": f"Cube solved! Total reward: {state['total_reward']:.3f}"}], state
        
        msg = f"Executed: {' '.join(moves)}\nReward: {state['reward']:.3f}"
        if predicted is None:
            msg += " (include state prediction for better reward)"
        msg += f"\n\nCurrent state:\n{new_cube.to_string()}\n\nProvide up to {info['max_moves']} moves in <move>...</move> and predict result in <state>...</state>."
        
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