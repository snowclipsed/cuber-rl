# cuber-rl

### Overview
- **Environment ID**: `cuber-rl`
- **Short description**: Multi-turn Rubik's cube solving environment with progressive reward shaping
- **Tags**: rubiks-cube, multi-turn, puzzle-solving, reinforcement-learning

### Datasets
- **Primary dataset(s)**: Procedurally generated scrambled cubes with configurable difficulty
- **Source links**: N/A (synthetic generation via `magiccube` and random scrambling)
- **Split sizes**: 1000 episodes (default), fully procedural so can be extended arbitrarily

### Task
- **Type**: Multi-turn
- **Parser**: XML tag parser (extracts moves from `<move>...</move>` tags)
- **Rubric overview**: 
  - Solving reward: 1.0 for solving + efficiency bonus $\min(1.0, \frac{d}{t})$ where $d$ is initial distance and $t$ is turns used
  - Progress reward: $\frac{\max(0, d - d')}{d}$ where $d'$ is distance after moves
  - Format penalty: 0.0 for invalid responses

The agent receives a scrambled Rubik's cube and must provide sequences of moves in Singmaster notation (U, D, L, R, F, B with optional ', 2 modifiers) to solve it. Each turn allows up to `max_moves_per_turn` moves. Rewards are given for making progress toward the solved state and bonus rewards for efficiency when solving.

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval cuber-rl
```

Configure model and sampling:

```bash
uv run vf-eval cuber-rl \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"scramble_ranges": [[4, 8], [9, 14]], "max_moves_per_turn": 3, "max_episode_steps": 20}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
- Easier scrambles (1-3 moves) are good for initial testing; harder scrambles (9-14 moves) require more strategic solving.

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `scramble_ranges` | List[Tuple[int, int]] | `[[4, 8], [9, 14]]` | List of (min, max) move ranges for scrambling. One range is randomly selected per episode. |
| `max_moves_per_turn` | int | `3` | Maximum number of moves the agent can execute per turn. |
| `max_episode_steps` | int | `20` | Maximum number of turns before episode terminates (actual max turns is `ceil(max_episode_steps / max_moves_per_turn)`). |

### Mechanics

**State Representation**: Cubes are displayed as unfolded nets showing all 6 faces (U=Top/White, L=Left/Orange, F=Front/Green, R=Right/Red, B=Back/Blue, D=Bottom/Yellow).

**Move Notation**: Standard Singmaster notation where single letters rotate clockwise 90°, apostrophe (') rotates counterclockwise 90°, and 2 rotates 180°.

**Response Format**: Agent must wrap moves in XML tags: `<move>U R' F2 D</move>`. Empty tags `<move></move>` indicate no moves (useful when cube is already solved).

**Reward Structure**:
- **Solving**: 1.0 base + efficiency bonus of $\min(1.0, \frac{\text{initial\_distance}}{\text{turns\_used}})$ 
- **Progress**: $\frac{\text{distance\_reduced}}{\text{initial\_distance}}$ for each turn (only positive progress counts)
- **Invalid format**: 0.0 reward for that turn

**Distance Metric**: Uses Kociemba's algorithm to compute optimal move count to solved state (cached for efficiency).

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Per-turn reward (progress or solving bonus) |
| `total_reward` | Cumulative reward across entire episode |
| `initial_dist` | Optimal moves needed from scrambled state |
| `final_distance` | Moves remaining to solve at episode end |
| `solved` | Boolean indicating if cube was solved |

### Example Interaction

```
Initial State (5 moves from solved):
        W W W
        W R W
        W W W
        
O O G   G G G   R O R   B B B
O O O   G W G   R R R   B Y B
O O O   G G G   R R R   B B B

        Y Y Y
        Y Y Y
        Y Y O

Agent: <move>U R' F</move>
Reward: 0.4 | Distance: 3
```

---

### Evaluation Reports

#### Model Performance (Pass@5, 10 puzzles episodes, Difficulty = 1 Move from Solved, No format reward)

| Model | Avg Reward / 2.0 | Solves / 50 | Equiv. % |
|-------|------------------|-------------|----------|
| GPT-5 | 1.76 | 44 | 88% |
| Claude Sonnet 4.5 | 0.60 | 15 | 30% |
| Claude Opus 4 | 0.36 | 9 | 18% |
| Gemini 2.5 Flash | 0.20 | 5 | 10% |
| Kimi k2 | 0.04 | 1 | 2% |
| Qwen-235B | 0.00 | 0 | 0% |

#### Performance by Scramble Difficulty (No Format Reward)

| Moves from Solved | GPT-5-nano | GPT-5-mini | GPT-5 |
|-------------------|------------|------------|-------|
| 1 | 0.920 | 1.120 | 1.200 |
| 2 | 0.020 | 0.120 | 1.220 |
| 3 | 0.033 | 0.000 | 1.125 |
| 4 | 0.010 | 0.000 | 0.820 |
| 5 | 0.000 | 0.000 | 0.500 |

GPT-5 maintains strong performance across all difficulty levels, while GPT-5-mini and GPT-5-nano show sharp degradation beyond simple 1-2 move scrambles, suggesting limited spatial reasoning capabilities for multi-step cube solving.