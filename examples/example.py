from activity_coordinator_agent.core.profiler import ActivityProfilerLogic, ResidentProfilerLogic
import dotenv
from google import genai
from pathlib import Path

dotenv.load_dotenv()

# client 
client = genai.Client()

# load patient and activities profile
activity_spec_path = Path(__file__).resolve().parent / "activities" / "activity_example_1.md"
resident_profile_path = Path(__file__).resolve().parent / "residents" / "resident_profile_example_1.md"

activity_spec_text = activity_spec_path.read_text(encoding="utf-8")
resident_profile_text = resident_profile_path.read_text(encoding="utf-8")

# run agents - profiler
activity_profiler_agent = ActivityProfilerLogic(client=client)
resident_profiler_agent = ResidentProfilerLogic(client=client)

activity_profile = activity_profiler_agent.run(activity_spec_text)
resident_profile = resident_profiler_agent.run(resident_profile_text)

# run agent - strategist
from activity_coordinator_agent.core.strategist import generate_strategy
strategy = generate_strategy(client, resident_profile, activity_profile)

print(strategy.model_dump_json(indent=2))