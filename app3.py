import streamlit as st
import pandas as pd


st.set_page_config(
    page_title="AutoML — Smart Prediction Tool",
    page_icon="🤖",
    layout="wide"
)

with st.sidebar:
    st.header("How this works")
    st.write(
        "1. Upload your data\n"
        "2. Pick what to predict\n"
        "3. We auto-detect and clean\n"
        "4. Models train & compete\n"
        "5. Get live predictions"
    )
    st.divider()
    st.caption("No coding or ML knowledge needed.")
    st.caption("Built for Smart India Hackathon 2026")

st.title("🤖 AutoML — Smart Prediction Tool")
st.markdown(
    "##### Upload your data. Get a trained model and live predictions — no coding required."
)
st.caption("Your uploaded data is used only for this session and is not stored permanently.")
st.divider()


# Detect classification or regression
def detect_problem_type(series):
    if series.dtype == "object":
        return "classification"

    unique = series.nunique()
    ratio = unique / len(series)

    if unique <= 20 or ratio < 0.05:
        return "classification"

    return "regression"


# Session state
if "trained" not in st.session_state:
    st.session_state["trained"] = False


st.subheader("Step 1 — Upload your data")
file = st.file_uploader("Upload CSV", type="csv")


if file:

    try:
        try:
            df = pd.read_csv(file, encoding="utf-8")
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, encoding="ISO-8859-1")

        if len(df) > 20000:
            st.warning(f"Dataset has {len(df):,} rows — using a random sample of 20,000 for faster training.")
            df = df.sample(n=20000, random_state=42)

        if df.empty:
            st.error("The uploaded CSV is empty.")
            st.stop()

        # Dataset information
        st.subheader("Step 2 — Preview your data")
        st.dataframe(df.head(), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Column types**")
            st.write(df.dtypes)
        with col2:
            st.write("**Missing values per column**")
            st.write(df.isnull().sum())

        st.write("**Basic stats**")
        st.dataframe(
            df.describe(include="all"),
            use_container_width=True
        )

        st.divider()

        # Target selection
        st.subheader("Step 3 — Select target column")

        target = st.selectbox(
            "Which column do you want to predict?",
            df.columns,
            index=len(df.columns) - 1
        )

        st.write(f"You selected: **{target}**")


        # Problem type
        st.subheader("Step 4 — Detected problem type")

        problem_type = detect_problem_type(df[target])

        st.write(
            f"This looks like a **{problem_type}** problem."
        )

        if problem_type == "classification":
            st.caption(
                "Reason: the target is text or has relatively few "
                "unique values, which is typical for categories/labels."
            )
        else:
            st.caption(
                "Reason: the target is numeric with many unique values, "
                "which is typical for continuous quantities."
            )


        problem_type = st.radio(
            "Not correct? You can override it:",
            [
                "Use detected type",
                "classification",
                "regression"
            ],
            index=0
        )

        if problem_type == "Use detected type":
            problem_type = detect_problem_type(df[target])

        st.divider()

        # Text columns
        st.subheader("Step 5 — Review text columns")

        text_cols = [
            col for col in df.columns
            if col != target and df[col].dtype == "object"
        ]

        if text_cols:

            st.write(
                "These text columns were found. Drop any that are "
                "just names/IDs rather than useful features:"
            )

            drop_cols = st.multiselect(
                "Drop these columns before training?",
                text_cols
            )

            if drop_cols:
                df = df.drop(columns=drop_cols)
                st.write(f"Dropped: {drop_cols}")

        st.divider()

        # Training
        st.subheader("Step 6 — Train and compare models")

        st.write(
            "This step cleans your data, handles missing values "
            "and encoding, trains several models, and ranks them."
        )


        if st.button("🚀 Run AutoML", type="primary"):

            with st.spinner(
                "Cleaning data, training models, and comparing results..."
            ):

                if problem_type == "classification":

                    from pycaret.classification import (
                        setup,
                        compare_models,
                        pull,
                        save_model
                    )

                else:

                    from pycaret.regression import (
                        setup,
                        compare_models,
                        pull,
                        save_model
                    )


                setup(
                    data=df,
                    target=target,
                    session_id=42,
                    verbose=False
                )

                best_model = compare_models()

                leaderboard = pull()

                save_model(
                    best_model,
                    "best_model"
                )


                # Save information for prediction
                st.session_state["trained"] = True
                st.session_state["problem_type"] = problem_type
                st.session_state["target"] = target
                st.session_state["features"] = [
                    col for col in df.columns
                    if col != target
                ]
                st.session_state["training_df"] = df.copy()


            st.success("Done! Here's how the models compared:")

            st.dataframe(
                leaderboard,
                use_container_width=True
            )


            st.download_button(
                "⬇️ Download leaderboard as CSV",
                leaderboard.to_csv(index=False),
                "leaderboard.csv",
                "text/csv"
            )


            st.write("**Best model selected:**")
            st.write(best_model)

        st.divider()

        # =========================================================
        # Prediction section
        # =========================================================

        if st.session_state.get("trained"):

            features = st.session_state["features"]
            training_df = st.session_state["training_df"]


            # Single prediction
            st.subheader("Step 7 — Try a single prediction")

            st.write(
                "Enter values below to get a live prediction "
                "from the best model."
            )

            input_data = {}

            for col in features:

                if training_df[col].dtype == "object":

                    options = (
                        training_df[col]
                        .dropna()
                        .unique()
                        .tolist()
                    )

                    input_data[col] = st.selectbox(
                        f"{col}",
                        options,
                        key=f"input_{col}"
                    )

                else:

                    mean = training_df[col].mean()

                    if pd.isna(mean):
                        mean = 0

                    input_data[col] = st.number_input(
                        f"{col}",
                        value=float(mean),
                        key=f"input_{col}"
                    )


            if st.button("🔮 Predict", type="primary"):

                if st.session_state["problem_type"] == "classification":

                    from pycaret.classification import (
                        load_model,
                        predict_model
                    )

                else:

                    from pycaret.regression import (
                        load_model,
                        predict_model
                    )


                model = load_model("best_model")

                input_df = pd.DataFrame([input_data])

                result = predict_model(
                    model,
                    data=input_df
                )

                st.success("Prediction complete:")

                st.dataframe(
                    result,
                    use_container_width=True
                )

            st.divider()

            # Batch prediction
            st.subheader("Step 8 — Or predict on many rows at once")

            st.write(
                "Upload a CSV with the same feature columns as above "
                "(everything except the target) to get predictions "
                "for every row."
            )

            st.write("Expected columns:")

            st.code(", ".join(features))


            batch_file = st.file_uploader(
                "Upload CSV for batch prediction",
                type="csv",
                key="batch"
            )


            if batch_file:

                try:

                    try:
                        batch_df = pd.read_csv(batch_file, encoding="utf-8")
                    except UnicodeDecodeError:
                        batch_file.seek(0)
                        batch_df = pd.read_csv(batch_file, encoding="ISO-8859-1")

                    missing = [
                        col for col in features
                        if col not in batch_df.columns
                    ]

                    if missing:

                        st.error(
                            f"Missing columns: {missing}"
                        )

                    else:

                        # Ignore extra columns
                        batch_df = batch_df[features]


                        if st.session_state["problem_type"] == "classification":

                            from pycaret.classification import (
                                load_model,
                                predict_model
                            )

                        else:

                            from pycaret.regression import (
                                load_model,
                                predict_model
                            )


                        model = load_model("best_model")

                        result = predict_model(
                            model,
                            data=batch_df
                        )


                        st.write("Predictions:")

                        st.dataframe(
                            result,
                            use_container_width=True
                        )


                        csv = (
                            result
                            .to_csv(index=False)
                            .encode("utf-8")
                        )

                        st.download_button(
                            "⬇️ Download predictions as CSV",
                            csv,
                            "predictions.csv",
                            "text/csv"
                        )


                except Exception as e:

                    st.error(
                        f"Couldn't process batch predictions: {e}"
                    )


    except Exception as e:

        st.error(
            "Something went wrong while processing the dataset."
        )

        st.exception(e)


else:

    st.info(
        "👆 Upload a CSV file above to get started."
    )