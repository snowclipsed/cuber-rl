import re
import json
import math
import random
from typing import List, Tuple

import verifiers as vf
from datasets import Dataset
from verifiers.types import Messages, State
from .solver import CubeSolver, generate_scramble
from .move import parse_moves, apply_moves, validate_moves
from .state import CubeState

def calculate_progress_reward(initial_state, final_state, num_moves, solver=None):
    """Reward for making progress toward solution"""
    if num_moves == 0:
        return 0
    
    if not solver:
        solver = CubeSolver()
    
    initial_dist = solver.get_distance(initial_state)
    final_dist = solver.get_distance(final_state)
    
    # Normalize by moves taken to prevent single-move gaming
    return (initial_dist - final_dist) / num_moves

def calculate_prediction_reward(predicted_state, actual_state, solver=None):
    """Reward for accurate mental modeling"""
    if predicted_state is None:
        return 0
    
    if predicted_state == actual_state:
        return 1.0
    
    if not solver:
        solver = CubeSolver()
    
    # Negative distance normalized by max possible distance (~20)
    distance = solver.get_distance(predicted_state, actual_state)
    return -distance / 20.0

def calculate_total_reward(initial_state, final_state, moves, predicted_state, 
                          progress_weight=0.7, prediction_weight=0.3, solver=None):
    """Combine progress and prediction rewards"""
    if not solver:
        solver = CubeSolver()
    
    progress = calculate_progress_reward(initial_state, final_state, len(moves), solver)
    prediction = calculate_prediction_reward(predicted_state, final_state, solver)
    
    return progress_weight * progress + prediction_weight * prediction

def calculate_step_rewards(initial_state, moves, predicted_state, solver=None):
    """All-in-one reward calculation for environment step"""
    if not solver:
        solver = CubeSolver()
    
    if not moves:
        return {'total': -1.0, 'progress': 0, 'prediction': 0, 'valid': False}
    
    try:
        final_state = apply_moves(initial_state, moves)
    except ValueError:
        return {'total': -1.0, 'progress': 0, 'prediction': 0, 'valid': False}
    
    progress = calculate_progress_reward(initial_state, final_state, len(moves), solver)
    prediction = calculate_prediction_reward(predicted_state, final_state, solver)
    
    return {
        'total': 0.7 * progress + 0.3 * prediction,
        'progress': progress,
        'prediction': prediction,
        'final_state': final_state,
        'valid': True
    }

def parse_llm_response(response_text):
    """Extract moves and predicted state from LLM response
    
    Args:
        response_text: LLM's response containing <move> and <state> tags
        
    Returns:
        tuple: (moves_list, predicted_state) or (None, None) if invalid
    """
    if not response_text:
        return None, None
    
    # Extract moves between <move> tags
    move_match = re.search(r'<move>(.*?)</move>', response_text, re.DOTALL)
    if not move_match:
        return None, None
    
    # Parse the move string into individual moves
    move_text = move_match.group(1).strip()
    moves = parse_moves(move_text)
    
    # Validate moves are legal Singmaster notation
    if not moves or not validate_moves(moves):
        return None, None
    
    # Extract predicted state between <state> tags
    state_match = re.search(r'<state>(.*?)</state>', response_text, re.DOTALL)
    predicted_state = None
    
    if state_match:
        try:
            predicted_state = CubeState(state_match.group(1).strip())
        except:
            # Invalid state format
            predicted_state = None
    
    return moves, predicted_state


def load_environment(
    difficulties: List[str] = ['easy', 'medium'],
    max_moves_per_turn: int = 3,
    dataset_size: int = 100,
    progress_weight: float = 0.7,
    prediction_weight: float = 0.3,
    seed: int = 42,
    **env_args,
) -> vf.Environment:
    
    random.seed(seed)
    solver = CubeSolver()
    
    def build_dataset() -> Dataset:
        data = []
        samples_per_difficulty = dataset_size // len(difficulties)
        
        for difficulty in difficulties:
            for _ in range(samples_per_difficulty):
                state, scramble, distance = generate_scramble(difficulty, solver)
                max_turns = math.ceil(distance / max_moves_per_turn)
                
                prompt = f"""You are solving a 3x3 Rubik's cube. Current state:

{state.to_string()}

Provide the next {max_moves_per_turn} moves (or fewer if solving) in Singmaster notation (U, D, L, R, F, B with optional ' for counter-clockwise or 2 for double).

Format your response as:
<move>YOUR_MOVES_HERE</move>
<state>PREDICTED_FINAL_STATE</state>

The predicted state should follow the same format as the input state."""
                
                data.append({
                    'prompt': [{'role': 'user', 'content': prompt}],
                    'answer': json.dumps({
                        'initial_state': state.to_string(),
                        'distance': distance,
                        'scramble': scramble
                    }),
                    'task': 'rubiks-cube',
                    'info': {
                        'state': state,
                        'distance': distance,
                        'max_turns': max_turns,
                        'turn_count': 0,
                        'history': []
                    }
                })
        
        return Dataset.from_list(data)
    
    class RubiksCubeEnv(vf.MultiTurnEnv):
        def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
            info = state.get('info', {})
            current_state = info.get('state')
            
            if current_state and current_state.is_solved():
                return True
            
            turn_count = info.get('turn_count', 0)
            max_turns = info.get('max_turns', 1)
            return turn_count >= max_turns
        
        def env_response(self, messages: Messages, state: State, **kwargs) -> Tuple[Messages, State]:
            # Get last assistant message
            last_assistant = None
            for msg in reversed(messages):
                if msg['role'] == 'assistant':
                    last_assistant = msg['content']
                    break
            
            if not last_assistant:
                return [{'role': 'user', 'content': 'Please provide your moves.'}], state
            
            # Parse response
            moves, predicted_state = parse_llm_response(last_assistant)
            
            # Update state
            info = state.get('info', {})
            current_state = info.get('state')
            
            if moves and current_state:
                try:
                    new_state = apply_moves(current_state, moves)
                    info['state'] = new_state
                    info['turn_count'] = info.get('turn_count', 0) + 1
                    info['history'].append({
                        'moves': moves,
                        'predicted': predicted_state,
                        'actual': new_state
                    })
                    
                    # Generate next prompt if not completed
                    if not new_state.is_solved() and info['turn_count'] < info['max_turns']:
                        next_prompt = f"""Current cube state after your moves:

{new_state.to_string()}

Distance to solution: {solver.get_distance(new_state)}

Provide the next {max_moves_per_turn} moves using the same format."""
                        return [{'role': 'user', 'content': next_prompt}], state
                    
                except ValueError:
                    return [{'role': 'user', 'content': 'Invalid moves. Try again.'}], state
            
            return [{'role': 'user', 'content': 'Task complete.'}], state
    
    def create_turn_reward(turn_idx):
        def turn_reward(completion, state, **kwargs):
            info = state.get('info', {})
            history = info.get('history', [])
            
            if turn_idx > len(history):
                return 0.0
            
            turn_data = history[turn_idx - 1]
            moves = turn_data.get('moves', [])
            predicted = turn_data.get('predicted')
            actual = turn_data.get('actual')
            
            if not moves:
                return -1.0
            
            # Calculate progress reward
            if turn_idx == 1:
                initial = info.get('state')
            else:
                initial = history[turn_idx - 2].get('actual')
            
            progress = calculate_progress_reward(initial, actual, len(moves), solver)
            
            # Calculate prediction reward
            prediction = calculate_prediction_reward(predicted, actual, solver)
            
            return progress_weight * progress + prediction_weight * prediction
        
        return turn_reward
    
    def overall_reward(completion, state, **kwargs):
        info = state.get('info', {})
        history = info.get('history', [])
        
        if not history:
            return 0.0
        
        total = 0.0
        for i in range(len(history)):
            reward_func = create_turn_reward(i + 1)
            total += reward_func(completion, state, **kwargs)
        
        # Bonus for solving
        final_state = history[-1].get('actual')
        if final_state and final_state.is_solved():
            total += 1.0
        
        return total / (len(history) + 1)
    
    # Create rubric
    max_turns_global = math.ceil(20 / max_moves_per_turn)
    reward_funcs = [overall_reward]
    weights = [1.0]
    
    rubric = vf.Rubric(funcs=reward_funcs, weights=weights)
    dataset = build_dataset()
    
    return RubiksCubeEnv(dataset=dataset, rubric=rubric, max_turns=max_turns_global)