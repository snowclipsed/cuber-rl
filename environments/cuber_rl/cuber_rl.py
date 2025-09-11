import re
import random
import math
from typing import Tuple, List, Optional, Dict
import numpy as np
import verifiers as vf
from verifiers.types import Messages, State
from datasets import Dataset
import magiccube
import kociemba

class CubeState:
    FACE_ORDER = ['U', 'L', 'F', 'R', 'B', 'D']
    FACE_NAMES = {'U': 'TOP', 'L': 'LEFT', 'F': 'FRONT', 'R': 'RIGHT', 'B': 'BACK', 'D': 'BOTTOM'}
    
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
            self.faces = {'U': 'W'*9, 'L': 'O'*9, 'F': 'G'*9, 'R': 'R'*9, 'B': 'B'*9, 'D': 'Y'*9}
    
    def to_string(self) -> str:
        return '\n'.join(f"{self.FACE_NAMES[f]}({f}): {self.faces[f][:3]}/{self.faces[f][3:6]}/{self.faces[f][6:9]}" 
                        for f in self.FACE_ORDER)
    
    def to_magiccube_string(self) -> str:
        """Convert to magiccube initialization string - just concatenate face colors"""
        result = ""
        for face in ['U', 'L', 'F', 'R', 'B', 'D']:
            result += self.faces[face]
        return result
    
    def from_magiccube_string(self, cube_string: str):
        """Parse magiccube's string format back to our representation"""
        if len(cube_string) != 54:
            raise ValueError(f"Invalid cube string length: {len(cube_string)}")
        
        idx = 0
        for face in ['U', 'L', 'F', 'R', 'B', 'D']:
            self.faces[face] = cube_string[idx:idx+9]
            idx += 9
    
    def to_kociemba(self) -> str:
        """Convert to kociemba format (different face order: U R F D L B)"""
        kociemba_order = ['U', 'R', 'F', 'D', 'L', 'B']
        kociemba_map = {'W': 'U', 'R': 'R', 'G': 'F', 'Y': 'D', 'O': 'L', 'B': 'B'}
        
        result = ""
        for face in kociemba_order:
            face_colors = self.faces[face]
            for color in face_colors:
                result += kociemba_map[color]
        return result
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return self.faces.copy()
    
    @classmethod
    def from_dict(cls, faces_dict: Dict):
        """Create from dictionary"""
        state = cls()
        state.faces = faces_dict
        return state
    
    def is_solved(self) -> bool:
        return all(len(set(colors)) == 1 for colors in self.faces.values())
    
    def __eq__(self, other) -> bool:
        return self.faces == other.faces if isinstance(other, CubeState) else False

def parse_moves(move_string: str) -> List[str]:
    """Parse move string into list of individual moves"""
    moves = re.findall(r"[UDLRFB][2']?", move_string.upper())
    return moves

def validate_moves(moves: List[str]) -> bool:
    """Check if moves are valid Singmaster notation"""
    valid_pattern = re.compile(r"^[UDLRFB][2']?$")
    return all(valid_pattern.match(m) for m in moves)

def apply_moves(state: CubeState, moves: List[str]) -> CubeState:
    """Apply moves to cube state using magiccube"""
    cube = magiccube.Cube(3, state.to_magiccube_string())
    
    for move in moves:
        cube.rotate(move)
    
    result_string = cube.get()
    
    new_state = CubeState()
    new_state.from_magiccube_string(result_string)
    return new_state

# Solver/Distance Module
class CubeSolver:
    def get_distance(self, state: CubeState, target: CubeState = None) -> int:
        """Get optimal move count to solve"""
        if target and not target.is_solved():
            return 20
        
        if state.is_solved():
            return 0
            
        try:
            kociemba_string = state.to_kociemba()
            solution = kociemba.solve(kociemba_string)
            if solution == "" or solution is None:
                return 0
            return len(solution.split())
        except Exception as e:
            print(f"Kociemba error: {e}")
            return 20
    
    def get_solution(self, state: CubeState) -> str:
        """Get solution moves"""
        try:
            if state.is_solved():
                return ""
            return kociemba.solve(state.to_kociemba())
        except:
            return ""

def generate_scramble(difficulty: str) -> CubeState:
    """Generate a scrambled cube"""
    ranges = {'easy': (1, 6), 'medium': (7, 14), 'hard': (15, 20)}
    min_moves, max_moves = ranges.get(difficulty, (1, 20))
    num_moves = random.randint(min_moves, max_moves)
    
    moves = []
    last_face = None
    opposite_faces = {'U': 'D', 'D': 'U', 'L': 'R', 'R': 'L', 'F': 'B', 'B': 'F'}
    
    for _ in range(num_moves):
        available = [f for f in 'UDLRFB' if f != last_face]
        if last_face and len(available) > 1:
            opp = opposite_faces.get(last_face)
            if opp in available and len(available) > 2:
                available.remove(opp)
        
        face = random.choice(available)
        modifier = random.choice(['', '2', "'"])
        moves.append(face + modifier)
        last_face = face
    
    state = CubeState()
    return apply_moves(state, moves)

# LLM Response Parser
def parse_llm_response(response: str) -> Tuple[Optional[List[str]], Optional[CubeState]]:
    """Parse moves and predicted state from LLM response"""
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

def prepare_rubiks_episode(x, difficulties=['easy', 'medium'], max_moves_per_turn=3, max_episode_steps=20):
    solver = CubeSolver()
    difficulty = random.choice(difficulties)
    cube_state = generate_scramble(difficulty)
    initial_distance = solver.get_distance(cube_state)
    max_turns = math.ceil(max_episode_steps / max_moves_per_turn)
    
    initial_msg = f"""You are solving a 3x3 Rubik's cube.

State representation: Each face shows its 9 stickers as a 3x3 grid (rows separated by /).
Colors: W=White, R=Red, B=Blue, O=Orange, G=Green, Y=Yellow

Current state:
{cube_state.to_string()}

Task: Provide up to {max_moves_per_turn} moves to make progress toward solving this cube.
- Use Singmaster notation: U, D, L, R, F, B (with optional ' for counterclockwise, 2 for double)
- Put your moves in <move>...</move> tags (use <move></move> if no moves needed)
- Stop early if cube becomes solved
- Predict the resulting state after your moves in <state>...</state> tags using the same format"""
    
    x["task"] = "rubiks-cube"
    x["info"] = {
        "cube_state_dict": cube_state.to_dict(),
        "initial_distance": initial_distance,
        "max_turns": max_turns,
        "max_moves_per_turn": max_moves_per_turn,
        "max_episode_steps": max_episode_steps,
        "difficulty": difficulty
    }
    x["prompt"] = [{"role": "user", "content": initial_msg}]
    return x

class RubiksCubeEnv(vf.MultiTurnEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.solver = CubeSolver()
    
    def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        cube_dict = state['info'].get('cube_state_dict')
        if cube_dict:
            cube_state = CubeState.from_dict(cube_dict)
            if cube_state.is_solved():
                return True
        return state['turn'] >= state['info'].get('max_turns', 10)
    
    def env_response(self, messages: Messages, state: State, **kwargs) -> Tuple[Messages, State]:
        info = state['info']
        
        # Process assistant's response
        if messages and messages[-1]['role'] == 'assistant':
            last_response = messages[-1]['content']
            moves, predicted = parse_llm_response(last_response)
            
            # Handle invalid or empty moves
            if moves is None:
                state['reward'] = -1.0
                state['done'] = True
                return [{"role": "user", "content": "Invalid format. Task terminated."}], state
            
            if len(moves) == 0:  # Empty moves allowed
                state['reward'] = 0
                state['done'] = False
                next_msg = f"""No moves executed.

Current state unchanged:
{CubeState.from_dict(info['cube_state_dict']).to_string()}

Provide up to {info['max_moves_per_turn']} moves to make progress."""
                return [{"role": "user", "content": next_msg}], state
            
            # Limit moves to max allowed
            moves = moves[:info['max_moves_per_turn']]
            
            # Apply moves
            cube_state = CubeState.from_dict(info['cube_state_dict'])
            initial_distance = self.solver.get_distance(cube_state)
            new_state = apply_moves(cube_state, moves)
            final_distance = self.solver.get_distance(new_state)
            
            # Calculate rewards
            progress = (initial_distance - final_distance) / len(moves)
            if predicted:
                pred_reward = 1.0 if predicted == new_state else -0.5
            else:
                pred_reward = -0.5
            
            state['reward'] = 0.7 * progress + 0.3 * pred_reward
            state['total_reward'] = state.get('total_reward', 0) + state['reward']
            
            # Update cube state
            info['cube_state_dict'] = new_state.to_dict()
            
            # Check if solved
            if new_state.is_solved():
                return [{"role": "user", "content": f"Cube solved! Reward: {state['total_reward']:.3f}"}], state
            
            # Continue with next turn
            next_msg = f"""Executed: {' '.join(moves)}
Progress reward: {progress:.3f}

Current state:
{new_state.to_string()}

Provide up to {info['max_moves_per_turn']} moves to make progress."""
            
            return [{"role": "user", "content": next_msg}], state
        
        return [], state

def load_environment(**kwargs) -> vf.Environment:
    '''Loads the Rubik's Cube RL environment'''
    dataset = Dataset.from_dict({'episode': range(1000)})
    
    # Map dataset to add required fields
    difficulties = kwargs.get('difficulties', ['easy', 'medium'])
    max_moves_per_turn = kwargs.get('max_moves_per_turn', 3)
    max_episode_steps = kwargs.get('max_episode_steps', 20)
    
    dataset = dataset.map(
        lambda x: prepare_rubiks_episode(x, difficulties, max_moves_per_turn, max_episode_steps)
    )
    
    parser = vf.Parser()
    
    def rubiks_reward_func(parser, completion, info, **kwargs) -> float:
        """Extract total reward from state"""
        state = kwargs.get('state', {})
        return state.get('total_reward', 0.0)
    
    rubric = vf.Rubric(parser=parser, funcs=[rubiks_reward_func])
    
    return RubiksCubeEnv(
        dataset=dataset,
        parser=parser,
        rubric=rubric,
        **kwargs
    )