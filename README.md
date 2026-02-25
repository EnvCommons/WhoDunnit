# Whodunnit? - Murder Mystery Environment

An OpenReward environment where AI agents solve murder mystery cases by gathering information and deducing the culprit, weapon, and location.

## Overview

Whodunnit is a deductive reasoning environment inspired by classic murder mystery puzzles. Agents must:
1. Gather information about suspects, weapons, locations, clues, and exhibits
2. Analyze the evidence to deduce who committed the murder
3. Determine what weapon was used and where the murder occurred
4. Submit their final answer for partial or full credit

## Features

- **Multiple Information Tools**: Agents can query suspects, weapons, locations, clues, motives, statements, and exhibits
- **Partial Reward Scoring**: Correct components earn partial credit (0.33 or 0.25 per component)
- **Flexible Task Format**: Supports optional motives, suspect statements, and exhibits with contextual clues
- **Provider Agnostic**: Works with OpenAI, Anthropic, Google Gemini, and OpenRouter

## Available Tools

### Information Gathering Tools (all return `finished=False`)

1. **list_suspects**: View all suspects with physical descriptions and features
2. **list_weapons**: View all potential murder weapons with weight classifications
3. **list_locations**: View all locations where the murder could have occurred
4. **list_clues**: View all clues and evidence found at the crime scene
5. **list_motives**: View potential motives (if available)
6. **list_statements**: View suspect statements (if available, note: murderer lies, others tell truth)
7. **view_exhibits**: View exhibits referenced in clues (e.g., Exhibit A: Logico's Detective Kit)

### Submission Tool

8. **submit_answer**: Submit final answer with `who`, `what`, `where`, and optionally `why`
   - Returns `finished=True`
   - Provides partial credit for partially correct answers
   - Shows which components were correct/incorrect

## Task Format

Each task in `tasks.json` contains:

```json
{
  "problem_name": "Murder in Hollywood",
  "problem_description": "Detective prompt text...",
  "suspects": [{"name": "...", "description": "...", "height": "...", ...}],
  "locations": [{"name": "...", "indoor": true, "description": "..."}],
  "weapons": [{"name": "...", "weight": "...", "description": "..."}],
  "clues": ["clue 1", "clue 2", ...],
  "motives": ["motive 1", ...],  // optional
  "statements": [{"suspect": "...", "statement": "..."}],  // optional
  "ground_truth": {
    "who": "The Amazing Aureolin",
    "what": "A Heavy Candle",
    "where": "The Enormous Bathroom",
    "why": "..."  // optional
  }
}
```

## Installation

### Local Development

```bash
cd whodunnit
pip install -r requirements.txt
python server.py
```

The server will start on `http://localhost:8080`.

### Docker

```bash
docker build -t whodunnit:latest .
docker run -p 8080:8080 whodunnit:latest
```

## Testing

### Local Test with OpenAI

```bash
export OPENAI_API_KEY=your_key_here
python test_agent.py
```

### Expected Behavior

For the "Murder in Hollywood" task:
- Agent gathers information using 4+ tool calls
- Deduces the correct answer: The Amazing Aureolin + A Heavy Candle + The Enormous Bathroom
- Receives 100% reward with "Perfect! You solved the case completely."

## Scoring

### 3-Component Tasks (no motive)
- WHO correct: +33.3%
- WHAT correct: +33.3%
- WHERE correct: +33.3%
- Total: 0-100%

### 4-Component Tasks (with motive)
- WHO correct: +25%
- WHAT correct: +25%
- WHERE correct: +25%
- WHY correct: +25%
- Total: 0-100%

## Example Agent Interaction

```
Turn 1: list_suspects()
  -> Returns: The Amazing Aureolin, Midnight III, Dame Obsidian with details

Turn 2: list_locations()
  -> Returns: The Enormous Bathroom, The Bedroom, The Screening Room

Turn 3: list_weapons()
  -> Returns: A Fork, An Aluminum Pipe, A Heavy Candle

Turn 4: list_clues()
  -> Returns: 5 clues including "body found in marble tub"

Turn 5: submit_answer(who="The Amazing Aureolin", what="A Heavy Candle", where="The Enormous Bathroom")
  -> Returns: 100% reward, finished=True
```

## Adding New Tasks

To add new murder mystery tasks:

### Option 1: Add to existing file
1. Edit `tasks_elementary.json` and add a new task object to the array
2. Include all required fields: `problem_name`, `problem_description`, `suspects`, `locations`, `weapons`, `clues`, `ground_truth`
3. Optionally include `motives` and `statements`
4. Ensure `ground_truth` has exact names matching suspects/weapons/locations

### Option 2: Create new task set
1. Create a new file following the pattern `tasks_*.json` (e.g., `tasks_advanced.json`, `tasks_expert.json`)
2. Format as a JSON array with task objects
3. The environment will automatically load all `tasks_*.json` files at startup
4. Task IDs are assigned sequentially across all loaded files

## Technical Details

- **Pattern**: AIME2025-style single-turn environment
- **Base Class**: `Environment` (no sandbox needed)
- **Data Loading**: Module-level JSON loading for fast access
- **Validation**: Pydantic models for task structure
- **Answer Comparison**: Case-insensitive with whitespace stripping
- **API**: Supports OpenAI Responses API, Anthropic Messages API, Google GenerateContent API

## Files

- `whodunnit.py` - Main environment class with all tools
- `server.py` - Minimal 8-line server wrapper
- `test_agent.py` - OpenAI Responses API test runner
- `tasks_elementary.json` - Elementary task data (currently 5 tasks: "Murder in Hollywood", "And Then There Was Another One", "The Art of the Kill", "The Last Train to Murder", and "Physician, Heal Myself!")
- `exhibits.json` - Exhibit data referenced in clues (e.g., Exhibit A: Logico's Detective Kit)
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container definition

**Note:** The environment loads all `tasks_*.json` files, allowing you to add additional task sets (e.g., `tasks_advanced.json`, `tasks_expert.json`) without modifying code.

## Deployment

This environment is deployed at:
- **Namespace**: `EnvCommons/whodunnit`
- **Repository**: https://github.com/EnvCommons/whodunnit

## Contributing

To contribute new murder mystery tasks:
1. Fork the repository
2. Add tasks to `tasks.json` following the existing format
3. Test with `test_agent.py`
4. Submit a pull request

## License

MIT License

## Credits

Inspired by classic murder mystery puzzle books and games.
