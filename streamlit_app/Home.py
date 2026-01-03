import streamlit as st
import dotenv
from google import genai

# Import your actual agent logic from src
from activity_coordinator_agent.core.profiler import ActivityProfilerLogic, ResidentProfilerLogic
from activity_coordinator_agent.core.strategist import generate_strategy
from activity_coordinator_agent.core.writer import generate_questions, edit_questions

# Load environment variables (API Key)
dotenv.load_dotenv()

st.set_page_config(page_title="Activity Coordinator", layout="wide")

def run_agent_logic(resident_text: str, activity_text: str):
    """
    Orchestrates the agent workflow using strings instead of files.
    """
    client = genai.Client()
    
    # 1. Profiling
    with st.status("Analyzing profiles...", expanded=True) as status:
        st.write("👤 Processing Resident Profile...")
        res_profiler = ResidentProfilerLogic(client=client)
        resident_profile = res_profiler.run(resident_text)
        
        st.write("🏃 Processing Activity Details...")
        act_profiler = ActivityProfilerLogic(client=client)
        activity_profile = act_profiler.run(activity_text)
        
        # 2. Strategy
        st.write("🧠 Generating Strategy...")
        strategy = generate_strategy(client, resident_profile, activity_profile)
        
        # 3. Drafting
        st.write("✍️ Drafting Questions...")
        drafts = generate_questions(client, resident_profile, strategy)
        
        # 4. Editing
        st.write("🎨 Polishing Output...")
        final_output = edit_questions(client, drafts)
        
        status.update(label="Workflow Complete!", state="complete", expanded=False)
        
    return final_output, strategy

# --- The UI ---

st.title("🤖 Activity Coordinator Agent")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Resident Profile")
    resident_input = st.text_area(
        "Paste resident details here...", 
        height=300,
        placeholder="Name: John Doe\nInterests: Gardening..."
    )

with col2:
    st.subheader("Activity Details")
    activity_input = st.text_area(
        "Paste activity details here...", 
        height=300, 
        placeholder="Activity: Morning Walk\nTime: 9 AM..."
    )

if st.button("🚀 Run Agent", type="primary", use_container_width=True):
    if not resident_input or not activity_input:
        st.warning("Please fill in both text boxes.")
    else:
        try:
            # Run the logic defined above
            final_result, strategy_result = run_agent_logic(resident_input, activity_input)
            
            st.divider()
            
            # Show Final Result
            st.header("✨ Final Questions")
            st.json(final_result.model_dump())
            
            # Show Strategy (Optional debugging view)
            with st.expander("View Underlying Strategy"):
                st.write(strategy_result.model_dump())
                
        except Exception as e:
            st.error(f"An error occurred: {e}")