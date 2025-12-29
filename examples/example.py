import datetime
import json
import argparse
import sys
from pathlib import Path
from typing import Optional, Union

import dotenv
from google import genai

from activity_coordinator_agent.core.profiler import ActivityProfilerLogic, ResidentProfilerLogic
from activity_coordinator_agent.core.strategist import generate_strategy
from activity_coordinator_agent.core.writer import generate_questions, edit_questions
from activity_coordinator_agent.schemas.activity_profile import ActivityProfile
from activity_coordinator_agent.schemas.resident_profile import ResidentProfile
from activity_coordinator_agent.schemas.strategist import ActivityStrategy
from activity_coordinator_agent.schemas.writer import DraftOutput, QuestionList


def get_latest_runtime_folder(base_path: Path) -> Optional[Path]:
    """Finds the latest timestamped folder in the base path."""
    if not base_path.exists():
        return None
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    # Assuming format YYYYMMDD_HHMMSS
    try:
        # Filter for folders that match our timestamp format roughly
        valid_subdirs = []
        for d in subdirs:
            try:
                datetime.datetime.strptime(d.name, "%Y%m%d_%H%M%S")
                valid_subdirs.append(d)
            except ValueError:
                continue
        
        if not valid_subdirs:
            return None
            
        latest = max(valid_subdirs, key=lambda x: x.name)
        return latest
    except ValueError:
        return None


def create_runtime_folder(base_path: Path) -> Path:
    """Creates a new timestamped folder."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = base_path / timestamp
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def run_example(
    resident_profile_path: Union[str, Path],
    activity_example_path: Union[str, Path],
    use_cache: Union[bool, str] = "auto"
) -> QuestionList:
    """
    Runs the activity coordinator workflow.
    
    Args:
        resident_profile_path: Path to the resident profile text file.
        activity_example_path: Path to the activity example text file.
        use_cache: 'auto', True, or False.
            - True: Read from latest cache, fail if missing.
            - False: Always start new run.
            - 'auto': Resume latest if exists, else start new.
    """
    dotenv.load_dotenv()
    client = genai.Client()
    
    resident_profile_path = Path(resident_profile_path).resolve()
    activity_example_path = Path(activity_example_path).resolve()
    
    # Data directory is sibling to this script + "data"
    base_data_path = Path(__file__).resolve().parent / "data"
    
    current_runtime_folder = None
    
    # Determine runtime folder
    if use_cache is True:
        current_runtime_folder = get_latest_runtime_folder(base_data_path)
        if not current_runtime_folder:
            raise FileNotFoundError(f"No cached run found in '{base_data_path}' and use_cache=True.")
        print(f"Using cached run: {current_runtime_folder}")
        
    elif use_cache == "auto":
        current_runtime_folder = get_latest_runtime_folder(base_data_path)
        if current_runtime_folder:
             print(f"Found cached run, checking for existing files in: {current_runtime_folder}")
        else:
             print("No cache found, starting new run.")
             current_runtime_folder = create_runtime_folder(base_data_path)
             
    else: # use_cache is False
        current_runtime_folder = create_runtime_folder(base_data_path)
        print(f"Starting new run: {current_runtime_folder}")

    # --- Helper to load or compute ---
    def load_or_compute(filename, compute_func, model_class, *args):
        filepath = current_runtime_folder / filename
        
        # In strict cache mode (True), we expect file to exist
        if use_cache is True:
            if filepath.exists():
                print(f"Loading {filename} from cache...")
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return model_class(**data)
            else:
                raise FileNotFoundError(f"Cache enabled but '{filename}' not found in {current_runtime_folder}")

        # In auto mode, try to load if exists
        if use_cache == "auto" and filepath.exists():
            print(f"Loading {filename} from cache...")
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                return model_class(**data)
        
        # Compute (for auto (missing file) or False)
        print(f"Computing {filename}...")
        result = compute_func(*args)
        
        # Save result
        with open(filepath, "w", encoding="utf-8") as f:
            print(f"Saving {filename}...")
            json.dump(result.model_dump(), f, indent=4)
            
        return result

    # 1. Profilers
    if not resident_profile_path.exists():
        raise FileNotFoundError(f"Resident profile not found: {resident_profile_path}")
    if not activity_example_path.exists():
        raise FileNotFoundError(f"Activity example not found: {activity_example_path}")

    resident_text = resident_profile_path.read_text(encoding="utf-8")
    activity_text = activity_example_path.read_text(encoding="utf-8")
    
    resident_profiler = ResidentProfilerLogic(client=client)
    activity_profiler = ActivityProfilerLogic(client=client)
    
    resident_profile = load_or_compute(
        "resident_profile.json", 
        resident_profiler.run, 
        ResidentProfile, 
        resident_text
    )
    
    activity_profile = load_or_compute(
        "activity_profile.json",
        activity_profiler.run,
        ActivityProfile,
        activity_text
    )

    # 2. Strategy
    # generate_strategy takes (client, resident, activity)
    strategy = load_or_compute(
        "strategy.json",
        lambda: generate_strategy(client, resident_profile, activity_profile),
        ActivityStrategy
    )
    
    # 3. Drafts (Writer)
    # generate_questions takes (client, resident, strategy)
    drafts = load_or_compute(
        "drafts.json",
        lambda: generate_questions(client, resident_profile, strategy),
        DraftOutput
    )
    
    # 4. Editor
    # edit_questions takes (client, drafts)
    edited_questions = load_or_compute(
        "edited_questions.json",
        lambda: edit_questions(client, drafts),
        QuestionList
    )
    
    return edited_questions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Activity Coordinator Agent example.")
    parser.add_argument("--resident", type=str, help="Path to resident profile")
    parser.add_argument("--activity", type=str, help="Path to activity example")
    parser.add_argument("--use-cache", type=str, choices=["true", "false", "auto"], default="auto", help="Cache usage strategy")
    
    args = parser.parse_args()
    
    # Convert string 'true'/'false' to booleans
    if args.use_cache.lower() == "true":
        cache_arg = True
    elif args.use_cache.lower() == "false":
        cache_arg = False
    else:
        cache_arg = "auto"
        
    # Default paths if not provided
    base_examples = Path(__file__).resolve().parent
    default_resident = base_examples / "residents" / "resident_profile_example_1.md"
    default_activity = base_examples / "activities" / "activity_example_1.md"
    
    resident_path = args.resident if args.resident else default_resident
    activity_path = args.activity if args.activity else default_activity
    
    try:
        final_output = run_example(resident_path, activity_path, use_cache=cache_arg)
        print("\n=== Final Output ===\n")
        print(final_output.model_dump_json(indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
