from activity_coordinator_agent.core.profiler import ActivityProfilerLogic, ResidentProfilerLogic
import dotenv
from google import genai
from pathlib import Path
import json
from activity_coordinator_agent.schemas.activity_profile import ActivityProfile
from activity_coordinator_agent.schemas.resident_profile import ResidentProfile
from activity_coordinator_agent.schemas.strategist import ActivityStrategy

dotenv.load_dotenv()

# client 
client = genai.Client()

# load patient and activities profile
# activity_spec_path = Path(__file__).resolve().parent / "activities" / "activity_example_1.md"
# resident_profile_path = Path(__file__).resolve().parent / "residents" / "resident_profile_example_1.md"

# activity_spec_text = activity_spec_path.read_text(encoding="utf-8")
# resident_profile_text = resident_profile_path.read_text(encoding="utf-8")

# # run agents - profiler
# activity_profiler_agent = ActivityProfilerLogic(client=client)
# resident_profiler_agent = ResidentProfilerLogic(client=client)

# activity_profile = activity_profiler_agent.run(activity_spec_text)
# resident_profile = resident_profiler_agent.run(resident_profile_text)

# with open(Path(__file__).resolve().parent / "activity_profile.json", "w", encoding="utf-8") as f:
#     json.dump(activity_profile.model_dump(), f, indent=4)
# with open(Path(__file__).resolve().parent / "resident_profile.json", "w", encoding="utf-8") as f:
#     json.dump(resident_profile.model_dump(), f, indent=4)

# # run agent - strategist
# from activity_coordinator_agent.core.strategist import generate_strategy
# strategy = generate_strategy(client, resident_profile, activity_profile)

# print(strategy.model_dump_json(indent=2))
# with open(Path(__file__).resolve().parent / "strategy.json", "w", encoding="utf-8") as f:
#     json.dump(strategy.model_dump(), f, indent=4)

# load json file
with open(Path(__file__).resolve().parent / "activity_profile.json", "r", encoding="utf-8") as f:
    activity_profile = ActivityProfile(**json.load(f))
with open(Path(__file__).resolve().parent / "resident_profile.json", "r", encoding="utf-8") as f:
    resident_profile = ResidentProfile(**json.load(f))
with open(Path(__file__).resolve().parent / "strategy.json", "r", encoding="utf-8") as f:
    strategy = ActivityStrategy(**json.load(f))

# run agent - writer
from activity_coordinator_agent.core.writer import generate_questions, edit_questions
drafts = generate_questions(client, resident_profile, strategy)
print(drafts.model_dump_json(indent=2))
with open(Path(__file__).resolve().parent / "drafts.json", "w", encoding="utf-8") as f:
    json.dump(drafts.model_dump(), f, indent=4)

# run agent - editor
edited_questions = edit_questions(client, drafts)
print(edited_questions.model_dump_json(indent=2))
with open(Path(__file__).resolve().parent / "edited_questions.json", "w", encoding="utf-8") as f:
    json.dump(edited_questions.model_dump(), f, indent=4)