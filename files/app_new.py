import streamlit as st

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to add a message to chat history
def add_message(sender, message):
    st.session_state.chat_history.append({"sender": sender, "message": message})

# Sidebar for chat history
st.sidebar.title("Chat History")
if st.session_state.chat_history:
    for chat in st.session_state.chat_history:
        if chat["sender"] == "You":
            st.sidebar.markdown(f"**🟦 You:** {chat['message']}")
        else:
            st.sidebar.markdown(f"**🟨 System:** {chat['message']}")
else:
    st.sidebar.write("No messages yet.")

# Main app for chatting
st.title("ChatGPT-like Chat App")
user_input = st.text_input("Your message:")

if st.button("Send"):
    if user_input.strip():
        # Add user's message to the history
        add_message("You", user_input.strip())
        # Simulate a response from the system (replace this with your actual logic)
        system_response = f"Echo: {user_input.strip()}"
        add_message("System", system_response)
        # Clear input box
        user_input = ""  # Automatically reset the input box

# Display the last system response for user reference
if st.session_state.chat_history:
    last_system_message = next(
        (chat["message"] for chat in reversed(st.session_state.chat_history) if chat["sender"] == "System"), ""
    )
    if last_system_message:
        st.subheader("Latest Response:")
        st.write(last_system_message)
