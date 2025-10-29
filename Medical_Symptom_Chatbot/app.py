import streamlit as st
from pipeline import MedicalChatbot

# -------------------------------
# STREAMLIT UI
# -------------------------------
def main():
    # Page configuration
    st.set_page_config(
        page_title="Medical Information Assistant",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Basic CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .emergency-warning {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
        font-weight: bold;
    }
    .chat-bubble {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
    }
    .user-bubble {
        background-color: #d1ecf1;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        text-align: right;
        border-right: 4px solid #17a2b8;
    }
    .info-box {
        padding: 10px;
        margin: 10px 0;
        border-radius: 8px;
        font-size: 0.9em;
    }
    .database-info {
        background-color: #e8f5e8;
        border-left: 4px solid #28a745;
    }
    .general-info {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .feature-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #dee2e6;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">Medical Information Assistant</div>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
    <strong>Important Disclaimer:</strong> This application provides general medical information for educational purposes only. 
    It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or 
    other qualified health provider with any questions you may have regarding a medical condition.
    </div>
    """, unsafe_allow_html=True)
    
    # Emergency warning
    st.markdown("""
    <div class="emergency-warning">
    For medical emergencies, call your local emergency number immediately! Do not rely on this application for emergency situations.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        with st.spinner("Loading medical database..."):
            st.session_state.chatbot = MedicalChatbot()
            st.session_state.messages = []
    
    # Sidebar
    with st.sidebar:
        st.title("Medical Assistant")
        
        st.markdown("### Information Sources")
        st.markdown("""
        This assistant can provide information from:
        - **Local Medical Database** (Your JSON files)
        - **General Medical Knowledge** (Gemini AI)
        """)
        
        st.markdown("### Ask About Anything")
        st.markdown("""
        You can ask about:
        - Any medication (prescription, OTC, herbal)
        - Medical conditions and symptoms
        - First aid and emergency procedures
        - Prevention and wellness
        - Drug interactions and side effects
        """)
        
        st.markdown("---")
        st.markdown("### Quick Access")
        
        # Medication quick access
        st.markdown("**Common Medications**")
        med_buttons = st.columns(2)
        common_meds = ["Ibuprofen", "Paracetamol", "Aspirin", "Amoxicillin", "Omeprazole", "Metformin"]
        for i, med in enumerate(common_meds):
            with med_buttons[i % 2]:
                if st.button(med, key=f"med_{med}"):
                    st.session_state.user_input = f"Tell me about {med} medication"
        
        # Conditions quick access
        st.markdown("**Common Conditions**")
        condition_buttons = st.columns(2)
        common_conditions = ["Migraine", "Hypertension", "Diabetes", "Asthma", "Arthritis", "Depression"]
        for i, condition in enumerate(common_conditions):
            with condition_buttons[i % 2]:
                if st.button(condition, key=f"cond_{condition}"):
                    st.session_state.user_input = f"Information about {condition}"
        
        st.markdown("---")
        st.markdown("### Database Stats")
        topics = st.session_state.chatbot.get_available_topics()
        st.write(f"Local topics: {len(topics)}")
        st.write(f"Indexed entries: {len(st.session_state.chatbot.index)}")
        st.write("General knowledge: Unlimited")
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### Ask Any Medical Question")
        
        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-bubble">
                    <strong>You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                # Add appropriate info box based on content
                if "database" in message["content"].lower():
                    info_class = "database-info"
                    info_text = "Information from medical database"
                else:
                    info_class = "general-info"
                    info_text = "General medical knowledge"
                
                st.markdown(f"""
                <div class="info-box {info_class}">
                    {info_text}
                </div>
                <div class="chat-bubble">
                    <strong>Assistant:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
        
        # User input with enhanced options
        user_input = st.text_area(
            "Your medical question:",
            value=st.session_state.get('user_input', ''),
            height=100,
            placeholder="Ask about any medication, condition, symptom, or health topic...",
            key="user_input_widget"
        )
        
        col1_1, col1_2, col1_3 = st.columns([1, 1, 1])
        
        with col1_1:
            if st.button("Smart Answer", type="primary", use_container_width=True):
                if user_input.strip():
                    # Add user message to chat
                    st.session_state.messages.append({"role": "user", "content": user_input})
                    
                    # Detect query type and use appropriate method
                    query_type = st.session_state.chatbot.detect_query_type(user_input)
                    
                    with st.spinner("Analyzing your question..."):
                        if query_type == "medication":
                            # Extract medication name
                            med_keywords = ['about', 'information on', 'tell me about']
                            med_name = user_input
                            for keyword in med_keywords:
                                if keyword in user_input.lower():
                                    med_name = user_input.lower().split(keyword)[-1].strip()
                                    break
                            response = st.session_state.chatbot.get_medication_info(med_name)
                        elif query_type == "condition":
                            # Extract condition name
                            cond_keywords = ['about', 'information on', 'tell me about']
                            cond_name = user_input
                            for keyword in cond_keywords:
                                if keyword in user_input.lower():
                                    cond_name = user_input.lower().split(keyword)[-1].strip()
                                    break
                            response = st.session_state.chatbot.get_condition_info(cond_name)
                        else:
                            response = st.session_state.chatbot.generate_response(user_input)
                    
                    # Add assistant response to chat
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Clear input
                    st.session_state.user_input = ""
                    st.rerun()
                else:
                    st.warning("Please enter a question")
        
        with col1_2:
            if st.button("Medication Info", use_container_width=True):
                if user_input.strip():
                    with st.spinner("Getting medication details..."):
                        response = st.session_state.chatbot.get_medication_info(user_input)
                    st.session_state.messages.append({"role": "user", "content": f"Medication: {user_input}"})
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.user_input = ""
                    st.rerun()
        
        with col1_3:
            if st.button("Condition Info", use_container_width=True):
                if user_input.strip():
                    with st.spinner("Getting condition details..."):
                        response = st.session_state.chatbot.get_condition_info(user_input)
                    st.session_state.messages.append({"role": "user", "content": f"Condition: {user_input}"})
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.user_input = ""
                    st.rerun()
    
    with col2:
        st.markdown("### Features")
        
        st.markdown("""
        <div class="feature-card">
        <strong>Any Medication</strong><br>
        Get information about prescription drugs, OTC medications, supplements
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
        <strong>Any Condition</strong><br>
        Learn about diseases, symptoms, treatments, and prevention
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
        <strong>Drug Interactions</strong><br>
        Understand medication safety and combinations
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
        <strong>Local + General</strong><br>
        Combines your database with general medical knowledge
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### Example Questions")
        st.markdown("""
        - "Tell me about Lipitor"
        - "Side effects of antidepressants"
        - "What is rheumatoid arthritis?"
        - "Can I take ibuprofen with blood pressure medication?"
        - "Symptoms of pneumonia"
        """)

if __name__ == "__main__":
    main()