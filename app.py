import streamlit as st
import pandas as pd

st.title("AutoML — Upload your data")
st.write("Upload a CSV file and let the app clean, train, and compare models for you.")

file = st.file_uploader("Upload CSV", type="csv")


def detect_problem_type(series: pd.Series) -> str:
    """Guess whether the target column is for classification or regression."""
    if series.dtype == "object":
        return "classification"
    unique_ratio = series.nunique() / len(series)
    if series.nunique() <= 20 or unique_ratio < 0.05:
        return "classification"
    return "regression"


if file:
    try:
        df = pd.read_csv(file)

        st.subheader("Preview")
        st.write(df.head())

        st.subheader("Column types")
        st.write(df.dtypes)

        st.subheader("Missing values per column")
        st.write(df.isnull().sum())

        st.subheader("Basic stats")
        st.write(df.describe())

        st.subheader("Select target column")
        target = st.selectbox(
            "Which column do you want to predict?",
            options=df.columns,
            index=len(df.columns) - 1  # defaults to last column
        )
        st.write(f"You selected: **{target}**")

        st.subheader("Detected problem type")
        problem_type = detect_problem_type(df[target])
        st.write(f"This looks like a **{problem_type}** problem.")

        if problem_type == "classification":
            st.caption(
                "Reason: the target column is either text, or has few unique "
                "values relative to the number of rows — typical of categories/labels."
            )
        else:
            st.caption(
                "Reason: the target column is numeric with many unique values — "
                "typical of a continuous quantity like price or temperature."
            )

        override = st.radio(
            "Not correct? You can override it:",
            options=["Use detected type", "classification", "regression"],
            index=0
        )
        if override != "Use detected type":
            problem_type = override
            st.write(f"Using **{problem_type}** instead.")

        # --- Let user drop text columns that are just names/IDs, not useful features ---
        st.subheader("Review text columns")
        text_cols = [col for col in df.columns if col != target and df[col].dtype == "object"]

        if text_cols:
            st.write(
                "These text columns were found. Drop any that are just names/IDs "
                "(like a car name or customer ID) rather than useful patterns:"
            )
            drop_cols = st.multiselect(
                "Drop these columns before training?",
                text_cols,
                default=[]
            )
            if drop_cols:
                df = df.drop(columns=drop_cols)
                st.write(f"Dropped: {drop_cols}")

        st.subheader("Train and compare models")
        st.write(
            "This step cleans your data (missing values, encoding) and trains "
            "several models automatically, then ranks them. This can take a "
            "minute or two depending on data size."
        )

        if st.button("Run AutoML"):
            with st.spinner("Cleaning data, training models, and comparing results..."):
                if problem_type == "classification":
                    from pycaret.classification import setup, compare_models, pull, save_model
                else:
                    from pycaret.regression import setup, compare_models, pull, save_model

                # setup() handles missing values + encoding internally
                setup(data=df, target=target, session_id=42, verbose=False)

                best_model = compare_models()
                leaderboard = pull()  # grabs the comparison table PyCaret just built

                # save model + remember problem type/target/feature columns for prediction step
                save_model(best_model, "best_model")
                st.session_state["problem_type"] = problem_type
                st.session_state["target"] = target
                st.session_state["feature_cols"] = [c for c in df.columns if c != target]
                st.session_state["df_for_dtypes"] = df  # to know numeric vs text per feature
                st.session_state["trained"] = True

            st.success("Done! Here's how the models compared:")
            st.dataframe(leaderboard)
            st.write("Best model selected:")
            st.write(best_model)

        # --- Prediction form: only shows up after training is done ---
        if st.session_state.get("trained"):
            st.subheader("Try a prediction")
            st.write("Enter values below to get a live prediction from the best model.")

            feature_cols = st.session_state["feature_cols"]
            ref_df = st.session_state["df_for_dtypes"]

            input_data = {}
            for col in feature_cols:
                if ref_df[col].dtype == "object":
                    # text/category column -> dropdown of existing values
                    options = ref_df[col].dropna().unique().tolist()
                    input_data[col] = st.selectbox(f"{col}", options=options, key=f"input_{col}")
                else:
                    # numeric column -> number input, default to column mean
                    default_val = float(ref_df[col].mean())
                    input_data[col] = st.number_input(
                        f"{col}", value=default_val, key=f"input_{col}"
                    )

            if st.button("Predict"):
                if st.session_state["problem_type"] == "classification":
                    from pycaret.classification import load_model, predict_model
                else:
                    from pycaret.regression import load_model, predict_model

                model = load_model("best_model")
                input_df = pd.DataFrame([input_data])
                result = predict_model(model, data=input_df)

                st.success("Prediction complete:")
                st.write(result)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
else:
    st.info("Upload a CSV file above to get started.")