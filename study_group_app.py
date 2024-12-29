import streamlit as st
import numpy as np
import sqlite3
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# 1. DATABASE SETUP
def init_db():

    conn = sqlite3.connect("study_groups.db")
    cursor = conn.cursor()

    # Create table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            gpa REAL,
            availability TEXT,   -- we'll store as comma-separated or JSON-like string
            subjects TEXT        -- we'll store as comma-separated string
        )
    """)
    conn.commit()
    return conn, cursor


def insert_student(name, gpa, availability_str, subjects_str, cursor, conn):
    cursor.execute("""
        INSERT INTO students (name, gpa, availability, subjects)
        VALUES (?, ?, ?, ?)
    """, (name, gpa, availability_str, subjects_str))
    conn.commit()


def fetch_all_students(cursor):
    cursor.execute("SELECT name, gpa, availability, subjects FROM students")
    rows = cursor.fetchall()
    return rows


# 2. HELPER FUNCTIONS
def convert_availability_to_vector(avail_list):
    """
    Convert a list of day-strings (e.g. ["Mon","Wed"]) to a 7-dimensional binary vector.
    """
    all_days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return [1 if day in avail_list else 0 for day in all_days]


def convert_subjects_to_vector(subject_list):
    all_subjects = ["Algorithms", "Data Structures", "Calculus", "Physics"]
    return [1 if subj in subject_list else 0 for subj in all_subjects]


def parse_csv_string(csv_string):
    if not csv_string:
        return []
    return csv_string.split(",")


# 3.STREAMLIT UI/LOGIC
def main():
    # Configure the Streamlit page
    st.set_page_config(
        page_title="Smart Study Group Finder",
        page_icon="🧠",
        layout="wide"
    )

    #initialize or connect to the SQLite database
    conn, cursor = init_db()

    st.title("Smart Study Group Finder")
    st.markdown("""
    Submit your info, then form groups based on availability, subjects, and GPA.
    """)

    #sidebar for extra controls
    st.sidebar.header("Database Info")
    if st.sidebar.button("Show All Records in DB"):
        rows = fetch_all_students(cursor)
        st.sidebar.write(f"Found {len(rows)} records:")
        for row in rows:
            st.sidebar.write(row)

    # ----------- COLLECT STUDENT DATA -----------
    with st.form("student_form", clear_on_submit=True):
        st.subheader("Enter Your Study Preferences")
        name = st.text_input("Name", placeholder="e.g., Alice Chen")
        gpa = st.number_input("GPA (2.0 to 4.0)", 2.0, 4.0, step=0.1)
        availability = st.multiselect(
            "Availability (select days)",
            ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        )
        subjects = st.multiselect(
            "Preferred Subjects",
            ["Algorithms", "Data Structures", "Calculus", "Physics"]
        )

        submitted = st.form_submit_button("Submit Data")
        if submitted:
            if not name.strip():
                st.error("Name cannot be empty.")
            else:
                #convert lists to comma-separated strings
                availability_str = ",".join(availability)
                subjects_str = ",".join(subjects)

                #insert into DB
                insert_student(name.strip(), gpa, availability_str, subjects_str, cursor, conn)
                st.success(f"Data for '{name}' submitted to database!")

    st.divider()

    # ----------- FORM GROUPS -----------
    if st.button("Form Groups"):
        # Fetch data from DB
        rows = fetch_all_students(cursor)
        if len(rows) < 2:
            st.warning("Need at least 2 students in the database to form groups.")
            return

        #prepare data for clustering
        names = []
        feature_vectors = []
        for (name_db, gpa_db, avail_db, subj_db) in rows:
            availability_list = parse_csv_string(avail_db)  # e.g. "Mon,Wed" -> ["Mon","Wed"]
            subjects_list = parse_csv_string(subj_db)  # e.g. "Algorithms,Calc" -> [...]

            availability_vec = convert_availability_to_vector(availability_list)
            subject_vec = convert_subjects_to_vector(subjects_list)
            data_vector = availability_vec + subject_vec + [gpa_db]

            names.append(name_db)
            feature_vectors.append(data_vector)

        #convert to numpy array
        X = np.array(feature_vectors)

        #scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        #prompt user for number of clusters
        k = st.slider("Number of Groups (Clusters)", 2, 10, 3)

        #k-Means clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)

        #display results
        st.subheader("Study Groups")
        groups = {}
        for i, label in enumerate(labels):
            groups.setdefault(label, []).append(names[i])

        for cluster_id, members in groups.items():
            st.markdown(f"**Group {cluster_id + 1}**")
            st.write(", ".join(members))

    # Close DB connection when app finishes
    # (Streamlit will rerun the script frequently, so usually we keep it open
    # but if you want to close it at the end of each run, you can.)
    # conn.close()


if __name__ == "__main__":
    main()
