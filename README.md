# Activity Coordinator Agent

An AI-powered system designed to assist care home activity coordinators in engaging residents with dementia. This tool generates personalized conversation starters and activity strategies by analyzing resident profiles and activity descriptions, ensuring interactions are meaningful, safe, and respectful.

## 🚀 Features

The agent operates through a multi-stage pipeline:

1.  **Profiler**: Analyzes unstructured text for Resident Profiles (biography, cognitive level, preferences) and Activity Descriptions.
2.  **Strategist**: Identifies "Semantic Bridges" (Hooks) to connect the resident's past history and interests to the current activity.
3.  **Writer**: Drafts context-aware conversation starters using specific engagement techniques (e.g., scaffolding, sensory focus).
4.  **Editor**: Reviews drafts for safety (avoiding grief triggers) and tone (removing "elderspeak" or patronizing language).

## 🛠️ Installation

This project requires Python 3.12+.

### Using `uv` (Recommended)

This project uses uv for dependency management.

```bash
# Install dependencies
uv sync
```

### Standard pip

```bash
pip install .
```

## ⚙️ Configuration

1.  Copy the example environment file:
    ```bash
    cp .env.example .env
    ```
2.  Add your Google Gemini API key to `.env`:
    ```
    GEMINI_API_KEY=your_api_key_here
    ```

## 🏃 Usage

An example script is provided to demonstrate the full workflow.

```bash
python examples/example.py
```

By default, it uses the example resident and activity profiles located in `examples/residents/` and `examples/activities/`. You can specify custom paths:

```bash
python examples/example.py \
  --resident path/to/resident_profile.md \
  --activity path/to/activity_description.md
```

## 📂 Project Structure

- **`src/activity_coordinator_agent/core/`**: Contains the logic for the different agents (Profiler, Strategist, Writer, Editor).
- **`src/activity_coordinator_agent/schemas/`**: Pydantic models defining the data structures for inputs and outputs.
- **`examples/`**: Contains the runner script and sample data.