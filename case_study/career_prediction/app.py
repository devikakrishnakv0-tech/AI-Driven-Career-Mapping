import streamlit as st
import pandas as pd
import joblib


# -------------------------------
# LOAD TRAINED MODEL
# -------------------------------

model = joblib.load("career_random_forest.pkl")

mlb = joblib.load("current_skills_mlb.pkl")

job_role_encoder = joblib.load("job_role_encoder.pkl")

feature_columns = joblib.load("model_features.pkl")
# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="Career Mapping and Skill-Based Job Role Prediction System ",
    page_icon="🎓",
    layout="wide"
)
# -------------------------------
# TITLE
# -------------------------------

st.title("Career Mapping and Skill-Based Job Role Prediction System ")

st.write(
    "Predict a suitable job role based on education, "
    "aptitude, skills and current abilities."
)
# -------------------------------
# STUDENT INFORMATION
# -------------------------------

st.header("Student Information")

student_name = st.text_input("Student Name")

education = st.selectbox(
    "Education",
    [
        "Bachelor's",
        "Diploma",
        "Other"
    ]
)

ug_specialization = st.text_input(
    "UG Specialization"
)

st.subheader("Master's Information")

masters_completed = st.selectbox(
    "Have you completed Master's?",
    [
        "Yes",
        "No"
    ]
)

if masters_completed == "Yes":

    masters_specialization = st.text_input(
        "Master's Specialization"
    )

else:

    masters_specialization = "Not Applicable"
    # -------------------------------
# SKILL AND APTITUDE RATINGS
# -------------------------------

st.header("Skill & Aptitude Ratings")

programming_skill = st.slider(
    "Programming Skill",
    0, 100, 50
)

mathematics_aptitude = st.slider(
    "Mathematics Aptitude",
    0, 100, 50
)

logical_reasoning = st.slider(
    "Logical Reasoning",
    0, 100, 50
)

analytical_thinking = st.slider(
    "Analytical Thinking",
    0, 100, 50
)

problem_solving = st.slider(
    "Problem Solving",
    0, 100, 50
)

communication = st.slider(
    "Communication",
    0, 100, 50
)

creativity = st.slider(
    "Creativity",
    0, 100, 50
)

teamwork = st.slider(
    "Teamwork",
    0, 100, 50
)

leadership = st.slider(
    "Leadership",
    0, 100, 50
)

business_knowledge = st.slider(
    "Business Knowledge",
    0, 100, 50
)

financial_knowledge = st.slider(
    "Financial Knowledge",
    0, 100, 50
)

design_skills = st.slider(
    "Design Skills",
    0, 100, 50
)
# -------------------------------
# CURRENT SKILLS
# -------------------------------

st.header("Current Skills")

available_skills = [
    "Python",
    "SQL",
    "Excel",
    "Power BI",
    "Machine Learning",
    "Communication",
    "Leadership",
    "Problem Solving",
    "Git",
    "Statistics",
    "Networking",
    "Research",
    "Teaching",
    "Figma",
    "Digital Marketing",
    "Financial Analysis",
    "Customer Service",
    "Recruitment",
    "Linux",
    "Data Visualization",
    "Database Management",
    "Cybersecurity",
    "HTML/CSS",
    "Java",
    "JavaScript",
    "Accounting",
    "Tally",
    "Project Management",
    "Writing",
    "Social Media",
    "SEO",
    "Cloud",
    "Sales",
    "Creativity",
    "Photoshop",
    "Lesson Planning",
    "Negotiation"
]

current_skills = st.multiselect(
    "Select your current skills",
    available_skills
)
# -------------------------------
# PREDICT BUTTON
# -------------------------------

st.divider()

predict_button = st.button(
    "🔮 Predict Career",
    type="primary"
)
# -------------------------------
# PREDICTION
# -------------------------------

if predict_button:

    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=feature_columns
    )
        # ---------------------------
    # NUMERICAL FEATURES
    # ---------------------------

    input_data["Programming Skill"] = programming_skill

    input_data["Mathematics Aptitude"] = mathematics_aptitude

    input_data["Logical Reasoning"] = logical_reasoning

    input_data["Analytical Thinking"] = analytical_thinking

    input_data["Problem Solving"] = problem_solving

    input_data["Communication"] = communication

    input_data["Creativity"] = creativity

    input_data["Teamwork"] = teamwork

    input_data["Leadership"] = leadership

    input_data["Business Knowledge"] = business_knowledge

    input_data["Financial Knowledge"] = financial_knowledge

    input_data["Design Skills"] = design_skills
        # ---------------------------
    # EDUCATION ENCODING
    # ---------------------------

    education_column = f"Education_{education}"

    if education_column in input_data.columns:
        input_data[education_column] = 1


    # UG Specialization

    ug_column = f"UG Specialization_{ug_specialization}"

    if ug_column in input_data.columns:
        input_data[ug_column] = 1


    # Master's completion

    masters_column = (
        f"Have you completed Master's?_{masters_completed}"
    )

    if masters_column in input_data.columns:
        input_data[masters_column] = 1


    # Master's specialization

    masters_spec_column = (
        f"Master's Specialization_{masters_specialization}"
    )

    if masters_spec_column in input_data.columns:
        input_data[masters_spec_column] = 1
            # ---------------------------
    # CURRENT SKILLS ENCODING
    # ---------------------------

    if current_skills:

        skill_encoded = mlb.transform(
            [current_skills]
        )

        skill_columns = [
            f"Skill_{skill}"
            for skill in mlb.classes_
        ]

        skill_df = pd.DataFrame(
            skill_encoded,
            columns=skill_columns
        )

        for column in skill_columns:

            if column in input_data.columns:

                input_data[column] = (
                    skill_df[column].iloc[0]
                )
                    # ---------------------------
    # MATCH TRAINING FEATURE ORDER
    # ---------------------------

    input_data = input_data[
        feature_columns
    ]
        # ---------------------------
    # RANDOM FOREST PREDICTION
    # ---------------------------

    prediction = model.predict(
        input_data
    )

    predicted_job_role = (
        job_role_encoder.inverse_transform(
            prediction
        )[0]
    )

    st.success(
        f"Predicted Job Role: {predicted_job_role}"
    )
        # ---------------------------
    # SKILL GAP ANALYSIS
    # ---------------------------

    role_skill_map = {

        "AI/ML Engineer": [
            "Python",
            "Machine Learning",
            "Statistics",
            "Git",
            "Problem Solving",
            "Research",
            "SQL",
            "Data Visualization"
        ],

        "Accountant": [
            "Excel",
            "Tally",
            "Accounting",
            "Financial Analysis",
            "Communication",
            "Data Visualization"
        ],

        "Business Analyst": [
            "Excel",
            "SQL",
            "Power BI",
            "Problem Solving",
            "Data Visualization",
            "Communication",
            "Research"
        ],

        "Cloud Engineer": [
            "Cloud",
            "Linux",
            "Networking",
            "Git",
            "Python",
            "SQL",
            "Problem Solving"
        ],

        "Content Writer": [
            "Writing",
            "Communication",
            "Creativity",
            "Research",
            "SEO",
            "Social Media"
        ],

        "Customer Support Executive": [
            "Customer Service",
            "Communication",
            "Problem Solving",
            "Leadership",
            "Research",
            "Excel"
        ],

        "Cybersecurity Analyst": [
            "Cybersecurity",
            "Networking",
            "Linux",
            "Python",
            "Problem Solving",
            "Research"
        ],

        "Data Analyst": [
            "Excel",
            "SQL",
            "Power BI",
            "Statistics",
            "Python",
            "Data Visualization",
            "Problem Solving"
        ],

        "Data Scientist": [
            "Python",
            "Machine Learning",
            "Statistics",
            "SQL",
            "Data Visualization",
            "Research",
            "Problem Solving"
        ],

        "Database Administrator": [
            "SQL",
            "Database Management",
            "Linux",
            "Networking",
            "Problem Solving",
            "Python"
        ],

        "Digital Marketing Specialist": [
            "Digital Marketing",
            "SEO",
            "Social Media",
            "Communication",
            "Research",
            "Writing",
            "Excel"
        ],

        "Financial Analyst": [
            "Financial Analysis",
            "Excel",
            "Statistics",
            "Power BI",
            "Accounting",
            "Data Visualization",
            "Research"
        ],

        "HR Executive": [
            "Recruitment",
            "Communication",
            "Leadership",
            "Customer Service",
            "Excel",
            "Research"
        ],

        "Project Manager": [
            "Project Management",
            "Leadership",
            "Communication",
            "Problem Solving",
            "Teamwork",
            "Research",
            "Excel"
        ],

        "Research Analyst": [
            "Research",
            "Statistics",
            "Excel",
            "SQL",
            "Data Visualization",
            "Problem Solving",
            "Writing"
        ],

        "Sales Executive": [
            "Customer Service",
            "Communication",
            "Negotiation",
            "Sales",
            "Digital Marketing",
            "Leadership"
        ],

        "Software Developer": [
            "Python",
            "Java",
            "Git",
            "Problem Solving",
            "SQL",
            "HTML/CSS",
            "JavaScript"
        ],

        "Teacher / Educator": [
            "Teaching",
            "Communication",
            "Lesson Planning",
            "Research",
            "Leadership",
            "Creativity"
        ],

        "UI/UX Designer": [
            "Figma",
            "Photoshop",
            "Creativity",
            "Communication",
            "Research"
        ],

        "Web Developer": [
            "HTML/CSS",
            "JavaScript",
            "Git",
            "Python",
            "SQL",
            "Web Development",
            "Problem Solving"
        ]
    }


    # Get required skills
    required_skills = set(
        role_skill_map.get(
            predicted_job_role,
            []
        )
    )


    # Student's current skills
    student_skills = set(
        current_skills
    )


    # Matched skills
    matched_skills = (
        required_skills
        .intersection(student_skills)
    )


    # Missing skills
    missing_skills = (
        required_skills
        .difference(student_skills)
    )


    # ---------------------------
    # DISPLAY SKILL GAP
    # ---------------------------

    st.subheader("📊 Skill Gap Analysis")


    st.write("### Required Skills")

    if required_skills:
        st.write(
            ", ".join(
                sorted(required_skills)
            )
        )
    else:
        st.write("No skill information available.")


    st.write("### Your Current Skills")

    if student_skills:
        st.write(
            ", ".join(
                sorted(student_skills)
            )
        )
    else:
        st.write("No current skills selected.")


    st.write("### Matched Skills")

    if matched_skills:
        st.success(
            ", ".join(
                sorted(matched_skills)
            )
        )
    else:
        st.write("No matched skills.")


    st.write("### Missing Skills / Skill Gap")

    if missing_skills:
        st.warning(
            ", ".join(
                sorted(missing_skills)
            )
        )
    else:
        st.success(
            "You already have all the required skills."
        )
            # ---------------------------
    # CAREER GUIDANCE
    # ---------------------------

    st.subheader("🎯 Career Guidance")

    if missing_skills:

        st.write(
            "To prepare for the predicted job role, "
            "you can focus on developing these skills:"
        )

        for skill in sorted(missing_skills):
            st.write(f"• {skill}")

    else:

        st.success(
            "You already have the required skills "
            "for this job role."
        )
