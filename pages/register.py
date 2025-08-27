# pages/register.py
import streamlit as st
import bcrypt
import os
from db import c, conn

st.set_page_config(
    page_title="User Registration"
)

st.title("📝 Register New User")

# --- Input Fields ---
username = st.text_input("Username")
email = st.text_input("Email")
phonenumber = st.text_input("Phone Number")
image = st.file_uploader("Upload Profile Image", type=["png","jpg","jpeg"])
password = st.text_input("Password", type="password")

# --- Register Button ---
if st.button("Register"):
    if not username or not email or not password:
        st.error("⚠ Please fill all required fields")
    else:
        # Save image locally
        image_path = ""
        if image:
            os.makedirs("profile_images", exist_ok=True)
            image_path = f"profile_images/{email}_{image.name}"
            with open(image_path, "wb") as f:
                f.write(image.getbuffer())

        # Hash password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        try:
            c.execute(
                "INSERT INTO users (username, email, phonenumber, image_path, password) VALUES (%s,%s,%s,%s,%s)",
                (username, email, phonenumber, image_path, hashed_password)
            )
            conn.commit()
            st.success("✅ Registration successful!")
            st.info("You can now log in.")
            st.balloons()
            st.switch_page("main.py")
        except Exception as e:
            st.error(f"⚠ Error: {e}")

# --- Link to Login for existing users ---
st.markdown("""
---
<p style='text-align:center; color:blue;'>
Already have an account?
</p>
""", unsafe_allow_html=True)
if st.button("Login Here"):
    st.switch_page("main.py")