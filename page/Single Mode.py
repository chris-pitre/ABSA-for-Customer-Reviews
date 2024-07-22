import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import re
from src.textutils import kmp_search
from src.models import load_setfit_model, load_pyabsa_model
import nltk

def setfit_prediction_display(review, preds):
    pretty_review = ""
    start_index = 0
    search_index = 0
    for pred in preds:
        color = ""
        match pred["polarity"]:
            case "positive":
                color = "green"
            case "neutral":
                color = "blue"
            case "negative":
                color = "red"
        search_index = kmp_search(pred["span"].lower(), review.lower(), search_index) + len(pred["span"])
        pretty_review += review[start_index:search_index].replace(pred['span'], f"**:{color}[{pred["span"]}]**", 1)
        start_index = search_index
    pretty_review += review[search_index:]
    return pretty_review

def pyabsa_prediction_display(json):
    pretty_review = ""
    for item in json:
        tokens = item["tokens"].copy()
        for i in range(len(item["aspect"])):
            match item["sentiment"][i]:
                case "Positive":
                    color = "green"
                case "Neutral":
                    color = "blue"
                case "Negative":
                    color = "red"
            for position in item["position"][i]:
                tokens[position] = f"**:{color}[{tokens[position]} ({item["confidence"][i]:.2%} confidence)]**"
        pretty_review += " ".join(tokens)
    return pretty_review

def single_mode():
    container = st.form("single_form")

    review = container.text_area("Enter a product review: ")
    col1, col2 = container.columns(2)
    split = col1.radio("Split up sentences when analyzing?", [True, False])
    detail = col2.radio("Which Model?", ["SetFit", "PyABSA"])
    detail = False if detail == "SetFit" else True

    if container.form_submit_button("Analyze"):
        if detail:
            more_detail(review, split, pyabsa_model)
        else:
            less_detail(review, split, setfit_model)

def more_detail(review, split, model):
    with st.spinner("Analyzing..."):
        if split:
            sentences = nltk.tokenize.sent_tokenize(review)
            atepc_result = []
            for sentence in sentences:
                atepc_result.append(model.extract_aspect(inference_source=[sentence], pred_sentiment=True)[0])
        else:
            atepc_result = model.extract_aspect(inference_source=[review], pred_sentiment=True)
        # in dataframe, probs the is total probability of each word [negative, neutral, positive]
        atepc_df = pd.DataFrame(atepc_result)
    st.write("## Full Text:")
    st.write(pyabsa_prediction_display(atepc_result))
    st.write("## Full Dataframe:")
    st.dataframe(atepc_df.explode(["aspect", "sentiment", "probs", "position", "confidence"]))
    if len(atepc_df["aspect"][0]) == 0:
        st.warning("No aspects found!")
        return
    st.write("### Proportion of Sentiments in the Review")
    num_preds = [0, 0, 0]
    for row in atepc_df["sentiment"]:
        for sentiment in row:
            match sentiment:
                case "Positive":
                    num_preds[0] += 1
                case "Neutral":
                    num_preds[1] += 1
                case "Negative":
                    num_preds[2] += 1
    fig, ax = plt.subplots()
    ax.pie(num_preds, labels=None, autopct="%1.1f%%", colors=["Green", "cornflowerblue", "Red"])
    ax.legend(labels=["Positive", "Neutral", "Negative"], bbox_to_anchor=(0, 1), loc='lower left', fontsize='small')
    num_preds = pd.DataFrame(num_preds)
    num_preds = num_preds.T
    num_preds.columns = ["Positive", "Neutral", "Negative"]
    col1, col2 = st.columns(2)
    col1.dataframe(num_preds)
    col2.pyplot(fig)
    st.write("### Detailed Confidence for each Sentiment in the Review")
    atepc_df_expand = atepc_df.explode(["aspect", "probs"])
    atepc_df_expand = atepc_df_expand[["aspect", "probs"]]
    aspects = atepc_df_expand["aspect"]
    atepc_df_expand = atepc_df_expand["probs"].apply(pd.Series).mul(100).round(2)
    atepc_df_expand.index = aspects
    atepc_df_expand.columns = ["Negative", "Neutral", "Positive"]
    atepc_df_expand = atepc_df_expand.dropna()
    st.dataframe(atepc_df_expand)
    fig, ax = plt.subplots()
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","cornflowerblue","green"])
    ax = atepc_df_expand.iloc[::-1].plot(kind="barh", stacked=True, colormap=cmap)
    ax.legend(ncols=atepc_df_expand.shape[0], bbox_to_anchor=(0, 1), loc='lower left', fontsize='small')
    for i, p in enumerate(ax.patches):
        if p.get_width() >= 5:
            ax.text(p.get_x() + p.get_width() / 2.0, p.get_y() + p.get_height() / 2.0, f"{p.get_width():.2f}%", fontsize="xx-small", ha="center", va="center")
    st.pyplot(ax.figure)


        
def less_detail(review, split, model):
    with st.spinner("Analyzing..."):
        st.write("## Full Text:")
        if split:
            review = re.split(r"(.+\.)", review)
            preds = model.predict(review)
            final_review = []
            for i in range(len(review)):
                final_review.append(setfit_prediction_display(review[i], preds[i]))
            final_review = "\n".join(final_review)
            st.write(final_review)
            preds = [pred for row in preds for pred in row]
        else:
            preds = model.predict(review)
            st.write(setfit_prediction_display(review, preds))
    col1, col2 = st.columns(2)
    preds_df = pd.DataFrame(preds)
    if preds_df.empty:
        st.warning("No aspects found!")
        return
    col1.write("### Aspects and Sentiments Dataframe")
    col1.dataframe(preds_df)
    num_preds = [
        preds_df[preds_df["polarity"] == "positive"].count()["span"], 
        preds_df[preds_df["polarity"] == "neutral"].count()["span"], 
        preds_df[preds_df["polarity"] == "negative"].count()["span"], 
    ]
    fig, ax = plt.subplots()
    ax.pie(num_preds, labels=None, autopct="%1.1f%%", colors=["Green", "cornflowerblue", "Red"])
    ax.legend(labels=["Positive", "Neutral", "Negative"], bbox_to_anchor=(0, 1), loc='lower left', fontsize='small')
    col2.write("### Proportion of Sentiments in the Review")
    col2.pyplot(fig)
    fig, ax = plt.subplots()
    ax.bar(["Positive"], num_preds[0], color="g")
    ax.bar(["Neutral"], num_preds[1], color="cornflowerblue")
    ax.bar(["Negative"], num_preds[2], color="r")
    col2.pyplot(fig)
 
st.write("# Aspect-Based Sentiment Analysis for Product Reviews")
plt.rcParams['font.family'] = 'sans-serif'
pyabsa_model = load_pyabsa_model()
setfit_model = load_setfit_model()
single_mode()