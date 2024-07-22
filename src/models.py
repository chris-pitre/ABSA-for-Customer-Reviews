import streamlit as st
from setfit import AbsaModel
import nltk
from pyabsa import ATEPCCheckpointManager
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from huggingface_hub import InferenceClient


@st.cache_resource
def load_setfit_model() -> AbsaModel:
    model = AbsaModel.from_pretrained(
        "./models/setfit-absa-model-aspect",
        "./models/setfit-absa-model-polarity"
    )
    return model

@st.cache_resource
def load_pyabsa_model():
    with st.spinner("Downloading model..."):
        nltk.download('punkt')
    aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(
        checkpoint='english',
        auto_device=True
    )
    return aspect_extractor

@st.cache_resource
def load_phi_model():
    model_name = "models/Phi-3-mini-4k-instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    return pipe, model, tokenizer

def build_messages(df1, df2, df3):
    messages = [{"role": "system", "content": """As an AI expert in market analysis for the beverage industry, you are tasked with evaluating various products from multiple companies. Conduct an Aspect-Based Sentiment Analysis on each product, focusing on key aspects such as Taste, Carbonation, Aftertaste, Aroma, Ingredients, Packaging, Price, Variety, Availability, Nutritional Content, Customer Service, Consistency and consumer sentiment areas for the product and not singular reviews. Assign a numerical score out of 10 for each aspect and also provide a consumer sentiment description for each aspect, reflecting the sentiment or performance of that aspect for the product.

        Additionally, perform a competitor comparison to gauge how each company's product stands in relation to others in the market using a numerical score for aspects as well as a qualitative analysis.

        Identify areas for improvement for each company's product, and based on your analysis, provide targeted recommendations for product innovation.
    """}]
    user_prompt = """
    Perform Aspect Based Sentiment Analysis for the following reviews:
    Here are some reviews about a product from company A:

    """
    df1_reviews = df1.iloc[:, 0].str.cat(sep="\n")
    user_prompt += df1_reviews
    user_prompt += """

    Here are some reviews about a competing product from company B:
    
    """
    df2_reviews = df2.iloc[:, 0].str.cat(sep="\n")
    user_prompt += df2_reviews
    user_prompt += """

    Here are some reviews from another competing product from company C:
    
    """
    df3_reviews = df3.iloc[:, 0].str.cat(sep="\n")
    user_prompt += df3_reviews
    new_message = {"role": "user", "content": user_prompt}
    messages.append(new_message)
    return messages


def phi_query(pipe, messages):
    client = InferenceClient("microsoft/Phi-3-mini-128k-instruct")
    result = client.chat_completion(messages, max_tokens=1024)
    return result