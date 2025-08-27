# main.py
import streamlit as st
import bcrypt
from db import c

st.set_page_config(
    page_title="User Login"
)

st.title("🔐 User Login")

email = st.text_input("Email")
password = st.text_input("Password", type="password")

if st.button("Login"):
    if not email or not password:
        st.error("⚠ Please enter email and password")
    else:
        c.execute("SELECT password, username FROM users WHERE email=%s", (email,))
        result = c.fetchone()

        if result and bcrypt.checkpw(password.encode('utf-8'), result[0].encode('utf-8')):
            st.session_state["authenticated"] = True
            st.session_state["username"] = result[1]
            st.success(f"✅ Welcome {result[1]}!")
            st.switch_page("pages/united.py")
        else:
            st.error("❌ Invalid email or password")

st.markdown("""
---
<p style='text-align:center; color:blue;'>
Don't have an account?
</p>
""", unsafe_allow_html=True)
if st.button("Register Here"):
    st.switch_page("pages/register.py")