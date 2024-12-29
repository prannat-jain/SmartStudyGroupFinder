import streamlit as st
import numpy as np
import sqlite3
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ------------------------
# 1. DATABASE SETUP
# ------------------------
def init_db():
    """
    Initialize a SQLite database with a 'students' table
    if it does not exist. Returns (connection, cursor).
    """
    conn = sqlite3.connect("study_groups.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            gpa REAL,
            availability TEXT,  -- e.g. "Mon,Wed,Fri"
            subjects TEXT       -- e.g. "Math,Computer Science"
        )
    """)
    conn.commit()
    return conn, cursor


def insert_student(name, gpa, availability_str, subjects_str, cursor, conn):
    """
    Insert a single student record into the 'students' table.
    """
    cursor.execute("""
        INSERT INTO students (name, gpa, availability, subjects)
        VALUES (?, ?, ?, ?)
    """, (name, gpa, availability_str, subjects_str))
    conn.commit()


def fetch_all_students(cursor):
    """
    Fetch all rows (name, gpa, availability, subjects) from 'students'.
    Returns a list of tuples.
    """
    cursor.execute("SELECT name, gpa, availability, subjects FROM students")
    rows = cursor.fetchall()
    return rows


# ------------------------
# 2. HELPER FUNCTIONS
# ------------------------
def convert_availability_to_vector(avail_list):
    """
    Convert a list of days (e.g. ["Mon", "Wed"]) into a 7-dim binary vector.
    """
    all_days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return [1 if day in avail_list else 0 for day in all_days]


def convert_subjects_to_vector(subject_list):
    """
    Convert a list of subjects (e.g. ["Math", "CS"]) into a binary vector
    for some defined list of all possible subjects.
    """
    # For demo, define a small subject list. Expand as needed.
    # Make sure this matches what's used in your form UI.
    all_subjects = [
        "Computer Science", "Mathematics", "Physics", "Chemistry",
        "Biology", "Economics", "History", "Literature",
        "Psychology", "Engineering"  # ... add more if you like
    ]
    return [1 if subj in subject_list else 0 for subj in all_subjects]


def parse_csv_string(csv_string):
    """
    Convert a comma-separated string (e.g. "Mon,Wed") back into ["Mon","Wed"].
    Handle empty strings safely.
    """
    if not csv_string:
        return []
    return csv_string.split(",")


def intersection_of_days(list_of_day_lists):
    """
    Return the intersection of day-lists.
    E.g. [["Mon","Wed"], ["Mon","Tue","Wed"]] -> ["Mon","Wed"].
    """
    if not list_of_day_lists:
        return []
    common_days = set(list_of_day_lists[0])
    for day_list in list_of_day_lists[1:]:
        common_days = common_days.intersection(day_list)
    return sorted(common_days)


def union_of_subjects(list_of_subject_lists):
    """
    Return the union of subject-lists.
    E.g. [["Math","Physics"], ["Physics","CS"]] -> ["CS","Math","Physics"].
    """
    all_subjs = set()
    for subj_list in list_of_subject_lists:
        all_subjs.update(subj_list)
    return sorted(all_subjs)


# ------------------------
# 3. FETCH FEATURES
# ------------------------
def fetch_student_features(names, cursor):
    """
    Given a list of student names, retrieve each student's availability & subjects & gpa,
    convert them to a numeric feature vector, and return a numpy array.
    Shape: (len(names), num_features).
    """
    feature_vectors = []

    for name in names:
        cursor.execute("""
            SELECT availability, subjects, gpa 
            FROM students
            WHERE name = ?
        """, (name,))
        row = cursor.fetchone()
        if row is None:
            # Not found or incomplete data
            feature_vectors.append([0] * 12)  # fallback vector
            continue

        availability_str, subjects_str, gpa_db = row
        availability_list = parse_csv_string(availability_str)  # e.g. "Mon,Wed" -> ["Mon","Wed"]
        subjects_list = parse_csv_string(subjects_str)  # e.g. "Math,CS" -> ["Math","CS"]

        availability_vec = convert_availability_to_vector(availability_list)
        subject_vec = convert_subjects_to_vector(subjects_list)

        # Example final vector: availability (7 dims) + subjects (len(all_subjects)) + 1 for GPA
        # Here, len(subject_vec) might be 10 if we have 10 subjects in all_subjects
        final_vector = availability_vec + subject_vec + [float(gpa_db)]
        feature_vectors.append(final_vector)

    return np.array(feature_vectors)


# ------------------------
# 4. SUBCLUSTERING LOGIC
# ------------------------
MAX_GROUP_SIZE = 5


def subdivide_large_cluster(cluster_members, cursor):
    """
    If cluster_members <= MAX_GROUP_SIZE, returns [cluster_members].
    Otherwise, we re-run K-Means on that subset,
    with k = ceil(len(cluster_members)/MAX_GROUP_SIZE).
    This yields sub-groups each of size <= 5.
    Returns a list of groups (each group is a list of names).
    """
    n_members = len(cluster_members)
    if n_members <= MAX_GROUP_SIZE:
        return [cluster_members]

    num_subclusters = math.ceil(n_members / MAX_GROUP_SIZE)

    # 1. Get feature vectors for these cluster_members
    X_sub = fetch_student_features(cluster_members, cursor)

    # 2. Re-run K-Means
    scaler_sub = StandardScaler()
    X_sub_scaled = scaler_sub.fit_transform(X_sub)

    kmeans_sub = KMeans(n_clusters=num_subclusters, random_state=42)
    labels_sub = kmeans_sub.fit_predict(X_sub_scaled)

    # 3. Build final sub-groups
    subgroups_dict = {}
    for i, name in enumerate(cluster_members):
        label = labels_sub[i]
        subgroups_dict.setdefault(label, []).append(name)

    # 4. Each sub-group might still be > 5 if the subcluster had wide distribution
    #    So we can recursively subdivide.
    final_subgroups = []
    for sg_members in subgroups_dict.values():
        if len(sg_members) > MAX_GROUP_SIZE:
            # recursively subdivide further
            final_subgroups.extend(subdivide_large_cluster(sg_members, cursor))
        else:
            final_subgroups.append(sg_members)

    return final_subgroups


# ------------------------
# 5. MAIN STREAMLIT APP
# ------------------------
def main():
    st.set_page_config(
        page_title="Smart Study Group Finder",
        page_icon="🧠",
        layout="wide"
    )

    conn, cursor = init_db()

    st.title("Smart Study Group Finder (Subclustering, Max Size = 5)")
    st.markdown("""
    1. Enter your info (availability, subjects, GPA).  
    2. Press **Form Groups** to see your assigned study group(s).  
    3. Large clusters automatically subdivide so no group exceeds 5 members.
    """)

    # ---- FORM FOR NEW STUDENT DATA ----
    with st.form("student_form", clear_on_submit=True):
        st.subheader("Enter Your Info")

        name = st.text_input("Name", placeholder="e.g., Alice Chen")
        gpa = st.number_input("GPA (2.0 to 4.0)", 2.0, 4.0, step=0.1)
        availability = st.multiselect(
            "Availability (select days)",
            ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        )

        # Larger subject list example
        possible_subjects = [
            "Computer Science", "Mathematics", "Physics", "Chemistry",
            "Biology", "Economics", "History", "Literature",
            "Psychology", "Engineering"
            # ... add more as needed
        ]
        subjects = st.multiselect(
            "Subjects (pick as many as apply)",
            possible_subjects
        )

        submitted = st.form_submit_button("Submit Data")
        if submitted:
            if not name.strip():
                st.error("Name cannot be empty.")
            else:
                # Convert to CSV strings for DB
                availability_str = ",".join(availability)
                subjects_str = ",".join(subjects)

                insert_student(name.strip(), gpa, availability_str, subjects_str, cursor, conn)
                st.success(f"Data for '{name}' submitted!")

    st.divider()

    # ---- FORM GROUPS BUTTON ----
    if st.button("Form Groups"):
        rows = fetch_all_students(cursor)
        if len(rows) < 2:
            st.warning("Need at least 2 students in the database to form groups.")
            return

        # Build feature matrix
        names = []
        feature_vectors = []
        for (name_db, gpa_db, avail_db, subj_db) in rows:
            names.append(name_db)
            # We won't convert everything here; let's do it after we pick out clusters
            # because we need a big X for initial K-Means

        # Step 1: Build the big feature matrix
        X = fetch_student_features(names, cursor)

        # Step 2: Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Step 3: Choose k (clusters) for the top-level grouping
        # (You could have a slider or a fixed number)
        k = st.slider("Number of main clusters (before subclustering)", 2, 10, 3)

        # Step 4: Top-level K-Means
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)

        # Collect members in each cluster
        main_clusters = {}
        for i, label in enumerate(labels):
            main_clusters.setdefault(label, []).append(names[i])

        # Step 5: Subdivide each cluster if it exceeds 5 members
        final_subgroup_list = []
        for cluster_id, members in main_clusters.items():
            subgroups = subdivide_large_cluster(members, cursor)
            final_subgroup_list.extend(subgroups)

        # Step 6: Display final subgroups, each with <= 5 members
        st.subheader("Final Groups (No More Than 5 Per Group)")
        for idx, subgroup_members in enumerate(final_subgroup_list, start=1):
            st.markdown(f"**Group {idx}**: {', '.join(subgroup_members)}")

            # (Optional) Show common availability & union of subjects
            group_avail_lists = []
            group_subj_lists = []
            for m in subgroup_members:
                # Re-fetch data from DB
                cursor.execute("""
                    SELECT availability, subjects FROM students WHERE name = ?
                """, (m,))
                row = cursor.fetchone()
                if row:
                    avail_str, subj_str = row
                    alist = parse_csv_string(avail_str)
                    slist = parse_csv_string(subj_str)
                    group_avail_lists.append(alist)
                    group_subj_lists.append(slist)

            common_days = intersection_of_days(group_avail_lists)
            all_subjs = union_of_subjects(group_subj_lists)

            st.write(f"- **Common Availability:** {', '.join(common_days) if common_days else 'None'}")
            st.write(f"- **All Subjects in Group:** {', '.join(all_subjs) if all_subjs else 'None'}")
            st.write("---")  # horizontal line


if __name__ == "__main__":
    main()
