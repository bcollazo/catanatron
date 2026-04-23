# LLM Player

The `LLMPlayer` is a Catan AI player that uses Large Language Models (LLMs) via Ollama to make game decisions. It sends the current game state and available actions to a local LLM instance and parses the response to select an action.

## Features

- **Network-based LLM calls**: Makes HTTP requests to a local Ollama instance for each decision
- **Templated prompts**: Uses a customizable markdown template with LangChain for prompt formatting
- **Simplified state representation**: Shows only essential game statistics to the LLM
- **Robust parsing**: Best-effort parsing of LLM responses with fallback to safe defaults

## Prerequisites

### 1. Install Ollama

Download and install Ollama from [https://ollama.ai/](https://ollama.ai/)

### 2. Install Python Dependencies

```bash
pip install langchain langchain-ollama
```

### 3. Pull an LLM Model

```bash
# Pull llama3.2 (recommended, fast and good quality)
ollama pull llama3.2

# Or try other models:
ollama pull mistral
ollama pull llama3.1
```

### 4. Start Ollama Server

```bash
ollama serve
```

The server will run on `http://localhost:11434` by default.

## Usage

### Basic Usage

```python
from catanatron import Game, RandomPlayer, Color
from catanatron.players.llm import LLMPlayer

# Create an LLM player
llm_player = LLMPlayer(Color.RED)

# Play against random opponents
players = [
    llm_player,
    RandomPlayer(Color.BLUE),
    RandomPlayer(Color.WHITE),
    RandomPlayer(Color.ORANGE),
]

game = Game(players)
winner = game.play()
print(f"Winner: {winner}")
```

### Custom Configuration

```python
# Use a different model and Ollama URL
llm_player = LLMPlayer(
    color=Color.RED,
    model_name="mistral",                    # Change the model
    ollama_base_url="http://localhost:11434" # Custom Ollama URL
)
```

## How It Works

### 1. State Extraction

On each decision, the player extracts simplified game state:
- Your stats: victory points, resources, dev cards, buildings available, bonuses
- Opponent stats: victory points, total resources, bonuses
- Game info: current turn, robber status

### 2. Prompt Formatting

The state is formatted using the template in `llm_player_prompt.md`:
- Shows your color and stats
- Lists opponents
- Numbers all available actions (0, 1, 2, ...)
- Asks LLM to respond with just the action number

### 3. LLM Call

The formatted prompt is sent to Ollama via HTTP request using LangChain.

### 4. Response Parsing

The LLM response is parsed to extract the action number:
- Looks for digits in the response
- Validates the number is in the valid range
- Falls back to action 0 if parsing fails

## Customizing the Prompt

You can customize the decision-making prompt by editing:
```
catanatron/catanatron/players/llm_player_prompt.md
```

The template uses LangChain's `PromptTemplate` format with variables like:
- `{color}` - Player's color
- `{your_vp}` - Victory points
- `{your_wood}`, `{your_brick}`, etc. - Resource counts
- `{actions_list}` - Numbered list of available actions
- And more...

## Performance Considerations

- **Latency**: Each decision requires a network call to Ollama, which can take 1-10 seconds depending on the model
- **Model choice**:
  - `llama3.2` - Good balance of speed and quality
  - `mistral` - Faster, good for testing
  - `llama3.1` - Slower but higher quality reasoning
- **Temperature**: Set to 0.7 by default for balanced exploration/exploitation

## Troubleshooting

### "ImportError: langchain is required"
Install dependencies:
```bash
pip install langchain langchain-ollama
```

### "Connection refused" or network errors
1. Check Ollama is running: `ollama serve`
2. Verify the URL: `curl http://localhost:11434`
3. Check firewall settings

### LLM returns invalid responses
The player has fallback logic:
- Logs a warning message
- Defaults to action 0 (first available action)
- Game continues without crashing

### Slow performance
- Use a smaller/faster model (e.g., `llama3.2` instead of `llama3.1`)
- Check Ollama's GPU acceleration is working
- Consider using a quantized model

## Example

See `examples/llm_player_example.py` for a complete working example.

## Architecture

```
LLMPlayer
├── llm.py              # Main player implementation
├── llm_player_prompt.md # Prompt template
└── README_LLM_PLAYER.md # This file
```

The implementation uses:
- **LangChain**: For prompt templating and LLM abstraction
- **langchain-ollama**: For Ollama integration
- **HTTP/REST**: Network calls to local Ollama instance
- **Regex parsing**: To extract action numbers from LLM responses
