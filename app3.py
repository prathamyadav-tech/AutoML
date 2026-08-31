import google.generativeai as genai
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AutoML — Smart Prediction Tool",
    page_icon="🤖",
    layout="wide"
)

# --- Gemini setup ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel("gemini-3.6-flash")
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False


# =========================================================
# Helper functions
# =========================================================

def detect_problem_type(series: pd.Series) -> str:
    """Guess whether the target column is for classification or regression."""
    if series.dtype == "object":
        return "classification"
    unique_ratio = series.nunique() / len(series)
    if series.nunique() <= 20 or unique_ratio < 0.05:
        return "classification"
    return "regression"


def explain_regression_result(r2: float, mae: float, target_name: str) -> str:
    """Plain-language summary of a regression model's quality (no LLM needed)."""
    if r2 >= 0.8:
        confidence = "high confidence"
    elif r2 >= 0.5:
        confidence = "moderate confidence"
    else:
        confidence = "low confidence"

    return (
        f"This model predicts **{target_name}** with **{confidence}** "
        f"(it explains about {r2 * 100:.0f}% of the pattern in your data). "
        f"On average, predictions are off by about **{mae:.2f}** — "
        f"use this as a helpful estimate, not an exact figure."
    )


def explain_classification_result(accuracy: float) -> str:
    """Plain-language summary of a classification model's quality (no LLM needed)."""
    if accuracy >= 0.9:
        return (
            f"This model is correct about **{accuracy * 100:.0f}%** of the time — "
            f"very reliable for decision-making."
        )
    elif accuracy >= 0.75:
        return (
            f"This model is correct about **{accuracy * 100:.0f}%** of the time — "
            f"good for guidance, but verify important decisions separately."
        )
    else:
        return (
            f"This model is correct about **{accuracy * 100:.0f}%** of the time — "
            f"treat predictions as rough guidance only, not a final answer."
        )


def explain_top_features(model, feature_cols):
    """Read feature importances directly off the model (no plotting, no crash risk)."""
    try:
        importances = model.feature_importances_
        pairs = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
        top = [name for name, score in pairs[:3] if score > 0]
        if top:
            return "The biggest factors driving this prediction were: **" + ", ".join(top) + "**."
        return None
    except AttributeError:
        try:
            coefs = model.coef_
            coefs = coefs[0] if len(coefs.shape) > 1 else coefs
            pairs = sorted(zip(feature_cols, abs(coefs)), key=lambda x: -x[1])
            top = [name for name, score in pairs[:3]]
            return "The biggest factors driving this prediction were: **" + ", ".join(top) + "**."
        except Exception:
            return None
    except Exception:
        return None


def explain_prediction(target, problem_type, input_data, predicted_value):
    """LLM explainer: what does THIS specific prediction mean and why is it useful."""
    inputs_text = ", ".join([f"{k} = {v}" for k, v in input_data.items()])

    prompt = f"""A user just got a prediction from a machine learning model.

They are predicting: "{target}" ({problem_type} problem)
The values they entered: {inputs_text}
The model's predicted result: {predicted_value}

Explain this result to the user in a short, clear way:

1. State plainly what the predicted value of "{target}" means, in context of the inputs they gave.
2. Explain why this number/category is useful to them — what decision or action it could inform.
3. Keep it to 3-4 sentences total, plain language, no statistics jargon, no headers or bullet points — just a short natural paragraph, as if a helpful assistant is explaining the result out loud."""

    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"(AI explanation unavailable right now: {e})"


def plot_prediction_context(training_df, target, problem_type, predicted_value):
    """Show where the new prediction falls compared to the training data."""
    fig, ax = plt.subplots(figsize=(6, 3))

    if problem_type == "regression":
        ax.hist(training_df[target].dropna(), bins=20, color="#1C7293", alpha=0.7)
        ax.axvline(predicted_value, color="#E8863C", linewidth=2.5, label="Your prediction")
        ax.set_xlabel(target)
        ax.set_ylabel("Number of records")
        ax.legend()
    else:
        counts = training_df[target].value_counts()
        colors = ["#E8863C" if str(cat) == str(predicted_value) else "#1C7293" for cat in counts.index]
        ax.bar(counts.index.astype(str), counts.values, color=colors)
        ax.set_ylabel("Count in training data")
        ax.tick_params(axis='x', rotation=45)

    ax.set_title("Where this prediction falls")
    fig.tight_layout()
    return fig


# =========================================================
# Sidebar
# =========================================================

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


# =========================================================
# Header
# =========================================================

st.title("🤖 AutoML — Smart Prediction Tool")
st.markdown(
    "##### Upload your data. Get a trained model and live predictions — no coding required."
)
st.caption("Your uploaded data is used only for this session and is not stored permanently.")
st.divider()


# =========================================================
# Session state
# =========================================================

if "trained" not in st.session_state:
    st.session_state["trained"] = False


# =========================================================
# Step 1 — Upload
# =========================================================

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

        # --- Step 2 — Preview ---
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
        st.dataframe(df.describe(include="all"), use_container_width=True)

        st.divider()

        # --- Step 3 — Target selection ---
        st.subheader("Step 3 — Select target column")
        target = st.selectbox(
            "Which column do you want to predict?",
            df.columns,
            index=len(df.columns) - 1
        )
        st.write(f"You selected: **{target}**")

        # --- Step 4 — Problem type ---
        st.subheader("Step 4 — Detected problem type")
        problem_type = detect_problem_type(df[target])
        st.write(f"This looks like a **{problem_type}** problem.")

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

        override = st.radio(
            "Not correct? You can override it:",
            ["Use detected type", "classification", "regression"],
            index=0
        )
        if override != "Use detected type":
            problem_type = override
            st.write(f"Using **{problem_type}** instead.")

        st.divider()

        # --- Step 5 — Review text columns ---
        st.subheader("Step 5 — Review text columns")
        text_cols = [col for col in df.columns if col != target and df[col].dtype == "object"]

        if text_cols:
            st.write(
                "These text columns were found. Drop any that are just names/IDs "
                "(like a car name or customer ID) rather than useful patterns:"
            )
            drop_cols = st.multiselect("Drop these columns before training?", text_cols, default=[])
            if drop_cols:
                df = df.drop(columns=drop_cols)
                st.write(f"Dropped: {drop_cols}")

        st.divider()

        # --- Step 6 — Train ---
        st.subheader("Step 6 — Train and compare models")
        st.write(
            "This step cleans your data (missing values, encoding) and trains "
            "several models automatically, then ranks them. This can take a "
            "minute or two depending on data size."
        )

        if st.button("🚀 Run AutoML", type="primary"):
            with st.spinner("Cleaning data, training models, and comparing results..."):
                if problem_type == "classification":
                    from pycaret.classification import setup, compare_models, pull, save_model
                else:
                    from pycaret.regression import setup, compare_models, pull, save_model

                setup(data=df, target=target, session_id=42, verbose=False)
                best_model = compare_models()
                leaderboard = pull()
                save_model(best_model, "best_model")

                st.session_state["problem_type"] = problem_type
                st.session_state["target"] = target
                st.session_state["feature_cols"] = [c for c in df.columns if c != target]
                st.session_state["df_for_dtypes"] = df
                st.session_state["trained"] = True
                st.session_state["leaderboard"] = leaderboard
                st.session_state["best_model_obj"] = best_model
                st.session_state["last_result"] = None
                st.session_state["last_explanation"] = None

            st.success("Done! Here's how the models compared:")
            st.dataframe(leaderboard, use_container_width=True)

            leaderboard_csv = leaderboard.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download leaderboard as CSV",
                leaderboard_csv,
                "leaderboard.csv",
                "text/csv"
            )

            # --- Level 1: Plain-language explanation (no LLM, always available) ---
            st.subheader("What this means, in plain terms")
            try:
                top_row = leaderboard.iloc[0]
                if problem_type == "regression":
                    r2 = float(top_row["R2"])
                    mae = float(top_row["MAE"])
                    st.info(explain_regression_result(r2, mae, target))
                else:
                    accuracy = float(top_row["Accuracy"])
                    st.info(explain_classification_result(accuracy))
            except Exception:
                st.caption("Summary unavailable — check the leaderboard table above for raw metrics.")

            # --- Level 2: Top features (no LLM, always available) ---
            feature_explanation = explain_top_features(best_model, st.session_state["feature_cols"])
            if feature_explanation:
                st.info(feature_explanation)

            st.write("**Best model selected:**")
            st.write(best_model)

        st.divider()

        # =========================================================
        # Prediction section
        # =========================================================

        if st.session_state.get("trained"):
            features = st.session_state["feature_cols"]
            training_df = st.session_state["df_for_dtypes"]

            # --- Step 7 — Single prediction ---
            st.subheader("Step 7 — Try a single prediction")
            st.write("Enter values below to get a live prediction from the best model.")

            input_data = {}
            for col in features:
                if training_df[col].dtype == "object":
                    options = training_df[col].dropna().unique().tolist()
                    input_data[col] = st.selectbox(f"{col}", options, key=f"input_{col}")
                else:
                    mean = training_df[col].mean()
                    if pd.isna(mean):
                        mean = 0
                    input_data[col] = st.number_input(f"{col}", value=float(mean), key=f"input_{col}")

            if st.button("🔮 Predict", type="primary"):
                if st.session_state["problem_type"] == "classification":
                    from pycaret.classification import load_model, predict_model
                else:
                    from pycaret.regression import load_model, predict_model

                model = load_model("best_model")
                input_df = pd.DataFrame([input_data])
                result = predict_model(model, data=input_df)

                # PyCaret names the prediction column differently across versions/problem types
                pred_col = next(
                    (c for c in ["prediction_label", "Label", "pred_label"] if c in result.columns),
                    result.columns[-1]
                )
                predicted_value = result.iloc[0][pred_col]

                st.session_state["last_result"] = result
                st.session_state["last_predicted_value"] = predicted_value
                st.session_state["last_input_data"] = input_data

                if GEMINI_AVAILABLE:
                    with st.spinner("Thinking..."):
                        st.session_state["last_explanation"] = explain_prediction(
                            st.session_state["target"],
                            st.session_state["problem_type"],
                            input_data,
                            predicted_value
                        )
                else:
                    st.session_state["last_explanation"] = None

            if st.session_state.get("last_result") is not None:
                st.success("Prediction complete:")
                st.dataframe(st.session_state["last_result"], use_container_width=True)

                if st.session_state.get("last_explanation"):
                    st.markdown("**What this means:**")
                    st.info(st.session_state["last_explanation"])
                elif not GEMINI_AVAILABLE:
                    st.caption("AI-powered explanation unavailable — Gemini API key not configured.")

                st.pyplot(plot_prediction_context(
                    training_df,
                    st.session_state["target"],
                    st.session_state["problem_type"],
                    st.session_state["last_predicted_value"]
                ))

            st.divider()

            # --- Step 8 — Batch prediction ---
            st.subheader("Step 8 — Or predict on many rows at once")
            st.write(
                f"Upload a CSV with the same feature columns as above "
                f"(everything except **{st.session_state['target']}**) to get predictions for every row."
            )
            st.write("Expected columns:")
            st.code(", ".join(features))

            batch_file = st.file_uploader("Upload CSV for batch prediction", type="csv", key="batch")

            if batch_file:
                try:
                    try:
                        batch_df = pd.read_csv(batch_file, encoding="utf-8")
                    except UnicodeDecodeError:
                        batch_file.seek(0)
                        batch_df = pd.read_csv(batch_file, encoding="ISO-8859-1")

                    missing = [col for col in features if col not in batch_df.columns]

                    if missing:
                        st.error(f"Missing columns: {missing}")
                    else:
                        batch_df = batch_df[features]

                        if st.session_state["problem_type"] == "classification":
                            from pycaret.classification import load_model, predict_model
                        else:
                            from pycaret.regression import load_model, predict_model

                        model = load_model("best_model")
                        result = predict_model(model, data=batch_df)

                        st.write("Predictions:")
                        st.dataframe(result, use_container_width=True)

                        csv = result.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇️ Download predictions as CSV",
                            csv,
                            "predictions.csv",
                            "text/csv"
                        )
                except Exception as e:
                    st.error(f"Couldn't process batch predictions: {e}")

    except Exception as e:
        st.error("Something went wrong while processing the dataset.")
        st.exception(e)

else:
    st.info("👆 Upload a CSV file above to get started.")