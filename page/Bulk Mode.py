import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
from src.textutils import kmp_search
from src.models import load_setfit_model, load_pyabsa_model, load_phi_model, build_messages, phi_query
import nltk
from nltk.tokenize import sent_tokenize
import json
import numpy as np

def more_detail():
    st.error("DISCLAIMER: This output was pre-computed on more powerful hardware and extracted here due to computational constraints on local machines.")
    file = open("./example_output.json")
    results = json.load(file)
    with st.expander("Prompt used for Phi-3 Mini"):
        messages = [
    {"role": "system", "content": """As an AI expert in market analysis for the beverage industry, you are tasked with evaluating various products from multiple companies. Conduct an Aspect-Based Sentiment Analysis on each product, focusing on key aspects such as Taste, Carbonation, Aftertaste, Aroma, Ingredients, Packaging, Price, Variety, Availability, Nutritional Content, Customer Service, Consistency and consumer sentiment areas for the product and not singular reviews. Assign a numerical score out of 10 for each aspect and also provide a consumer sentiment description for each aspect, reflecting the sentiment or performance of that aspect for the product.

Additionally, perform a competitor comparison to gauge how each company's product stands in relation to others in the market using a numerical score for aspects as well as a qualitative analysis.

Identify areas for improvement for each company's product, and based on your analysis, provide targeted recommendations for product innovation.

Please present your findings in a structured JSON format, suitable for conversion into a dataframe. The JSON should include an array of objects, each representing a company's product analysis, with nested objects for aspects, company performance, competitor comparison, and areas for improvement.

Here is the structure for the JSON output:

[
  {
    "company": "Specify the company name",
    "product": "Specify the product name",
    "aspects": {
      //specify aspects: {
        //specify sub aspects : Score
        // include other sub aspects as needed
      },
      // specify other aspects in the same format as needed

    "aspects described": {
      //describe aspect sentiment:
      "aspect name": sentiment description based on reviews,
      // add aspects as needed
    },
    "
    "competitor_comparison_described": {
      //Other Company : description of its comparison to our company
      // Include other competitors as needed
    },
    "areas_for_improvement": {
      "Company Name": {
        "aspect": "Suggested area for improvement with how much to improve the metric",
        // Include other aspects as needed
      }
      // Include other companies as needed
    }
    // Include additional products as needed
  }
  // Include additional companies as needed
]
"""},
    {"role": "user", "content": """Perform the given task for the following reviews:
    Here are some reviews about a product from company A:

I've been a loyal fan of this soda for years, primarily because of its distinct taste that I haven't found anywhere else. The balance of sweetness and spice is just right, making it incredibly satisfying, especially on a hot day. However, I've become more health-conscious over time, and the high content of artificial ingredients is starting to concern me. Additionally, the packaging design could be improved. It's often difficult to open, and I feel like they haven't kept up with more eco-friendly trends, which is important to me.
The aftertaste lingers a bit longer than I'd like, but the initial flavor burst is worth it. It's a good value for the price, especially when you catch it on sale.
I'm not a fan of the packaging. It's hard to open sometimes, but the soda itself is delicious. It's my go-to drink for a quick pick-me-up.
It's readily available, which I appreciate. The taste is fantastic, but I'm becoming more aware of the ingredients and their impact on my health.
The taste has been consistent over the years, which is great, but I'm starting to look for drinks with better ingredients.
Great soda with a consistent taste. Wish the packaging was better and the ingredient list shorter.
I appreciate the effort to cut down on calories, but the artificial sweeteners leave a noticeable aftertaste.
It's got a good fizz, decent price, but the aftertaste and ingredients are a letdown. I do like the lower calorie count, though.
The taste is one of a kind, and the lower calorie content is a bonus, but the packaging isn't the easiest to recycle.
I'm torn. The taste is excellent, and it's affordable, but the ingredients list and the aftertaste leave much to be desired.
I appreciate that it has fewer calories than many other sodas on the market, which helps me enjoy it without too much guilt. That said, the aftertaste can be a bit overpowering and tends to linger longer than I'd like.
I'm not a fan of the packaging. It's hard to open sometimes, but the soda itself is delicious. It's my go-to drink for a quick pick-me-up.
This soda has been a staple in my household for its taste and affordability. It's the kind of drink that seems to go well with any meal and any occasion. The carbonation is perfectly dialed in, providing that satisfying fizz without being too aggressive. On the downside, the ingredient list leaves a lot to be desired, with more artificial additives than I'm comfortable with. I've also noticed that the packaging, while distinctive, hasn't changed much over the years and could benefit from a more modern, sustainable approach
I've tried to get behind this soda because of its unique taste, but the more I drink it, the more the artificial sweeteners bother me. There's an aftertaste that lingers far too long, making the overall experience rather unpleasant. On top of that, the ingredients list is a bit concerning with all the additives and preservatives. I'm all for enjoying a good soda, but I think it's important to be mindful of what's inside the can. Unfortunately, this brand just doesn't meet my expectations on either front.
The taste is fantastic, and it has that signature fizz that I love.
I was initially drawn to this soda because of its lower calorie promise, but I must say, I'm quite disappointed. The taste is overshadowed by an unpleasant aftertaste that just doesn't sit well with me. Additionally, the packaging is a letdown. Not only does it feel cheap and fragile, but it's also not environmentally friendly, which is something I'm increasingly concerned about. I had hoped for a better experience, but between the questionable ingredients and the subpar packaging, I think I'll be looking for my fizzy fix elsewhere.
I honestly can't understand the appeal of this soda. The flavor is just off to me—it tastes artificial and nothing like the description. I've tried it cold, over ice, nothing helps. It's a hard pass for me.


Here are some reviews about a competing product from company B

The price is fair, and the taste is consistent, but I'm always left feeling guilty about the sugar content.
The taste is classic and always refreshing, but I'm trying to watch my calorie intake, and this doesn't help. The packaging is sleek and practical, though.
The sweetness is a bit much for me. I enjoy the occasional can, but I'm trying to be more health-conscious.
The clean aftertaste and simple ingredients make this my preferred soda. I just wish they had more low-calorie options.
This soda's taste is consistent, which I appreciate. The high calorie and sugar content, though, prevent me from enjoying it more often.
I've been drinking this soda for years, and the taste has remained consistent. It's always my first choice when I need something fizzy
The classic taste is there, but I'm not impressed with the variety of flavors available. It would be nice to see some new innovations in their lineup.
The fewer ingredients are a plus, but I'm concerned about the sugar and calorie content. The aftertaste is quite good, though.
The variety isn't vast, but the main flavor is so good that I don't mind too much.
The aroma is mild and the ingredients list is reassuringly short, which I appreciate in a beverage
I just can't get behind the taste of this soda. It's too syrupy and sweet for my palate. I've tried it on different occasions, thinking maybe it would grow on me, but it just hasn't. I'll be avoiding this one in the future.
I was unimpressed with the lack of flavor variety and the overly sweet aftertaste. The packaging is decent, but that's not enough to win me over.
The ingredients list is shorter than some, but that doesn't excuse the excessive sweetness and high calorie content. The aftertaste is also unpleasant
It's a standard soda with a typical sugary taste and decent carbonation. The packaging is fine, and the price is in line with other similar options on the market.
I've been drinking this particular soda for as long as I can remember. It's the kind of drink that seems to define what a classic soda should taste like. The consistency in flavor from bottle to bottle is impressive, and the aftertaste is just right – not too sweet, not too bitter. However, the high calorie and sugar content is becoming a bigger issue for me as I try to adopt a healthier lifestyle. While I enjoy the taste and the nostalgia it brings, I'm increasingly looking for alternatives that offer a similar experience without the health drawbacks. The packaging is a positive aspect, though, as it's clear the company has put thought into environmental impact.


Here are some reviews from another competing product from company C:


As someone who enjoys a wide variety of teas, I find this brand to be a standout for its diverse flavor offerings and commitment to using natural ingredients. Each flavor is unique and well-crafted, providing a refreshing alternative to the typical soda. The packaging is not only visually appealing but also practical, making it easy to store and transport. However, I've had some difficulty finding certain flavors in stores, which can be frustrating. The aftertaste is generally pleasant, although some flavors have a slightly bitter finish that I think could be improved upon. Despite these minor issues, it's a brand I regularly recommend to friends and family.
The taste is good, and the packaging is some of the best I've seen. It's a shame it's not more widely available.
The variety keeps me coming back, and the ingredients feel healthier than other options. The aftertaste is a bit strong, though.
The drink is usually easy to find, and the nutritional content is decent for a sweetened beverage.
The drink leaves a bad aftertaste, and the lack of availability is a major downside. It's not something I'll be seeking out again.
The packaging design is modern and appealing, but the drink itself doesn't always deliver on taste. I've also experienced some inconsistency with the product's quality.
While the price and packaging are positives, the taste has been a bit inconsistent lately. I'm also not thrilled about having to hunt for it at different stores.
The drink offers a decent variety, but I find the flavor to be a bit too mild. Additionally, the cans sometimes arrive dented, which is concerning.
I love the commitment to natural ingredients and the variety they offer.
The price is great, but you get what you pay for with a taste that doesn't quite hit the mark and an aftertaste that's less than pleasant. The ingredients are a mixed bag, and the availability is inconsistent, which adds to the frustration.
For the price, you can't beat the quality of this beverage. The taste is consistently good across the different flavors, and while the aftertaste is noticeable, it's actually quite pleasant and not overpowering. The packaging is sturdy and environmentally friendly, which is a huge plus for me.
The diverse range of flavors keeps my taste buds excited, and the natural ingredients make me feel good about my drink choice. Even though it's not always in stock, the hunt makes it feel like a hidden gem. The packaging is not only stylish but also functional, making it easy to enjoy on the go.
The packaging is visually appealing, and the ingredients are top-notch. Availability could be better.
Refreshing taste, but I wish it was easier to find. The variety is good when you can get it.
I really wanted to like this tea, but the taste is just not for me. It's either too bland or too overpowering, depending on the flavor. I've tried several, hoping to find one that suits me, but I've been disappointed every time. I won't be buying this again.

"""},
]
        st.write(messages)
    with st.expander("Phi-3 Mini exact output"):
        st.write(results)
    global_avg_score = 0
    max_score = 0
    for company in results:
        for aspect in company["aspects"]:
            max_score += 1
            global_avg_score += company["aspects"][aspect]["score"] 
    global_avg_score /= max_score

    for company in results:
        st.write(f"## {company["company"]}: {company["product"]}")
        table = "| Aspect            | Rating | Description                                                                 |\n|-------------------|--------|-----------------------------------------------------------------------------|\n"
        total_score = 0
        max_score = 0
        score_list = []
        count = 0
        for aspect in company["aspects"]:
            table += f"| **{aspect}** | {company["aspects"][aspect]["score"]}/10 | {company["aspects"][aspect]["consumer_sentiment"]} |\n"
            total_score += company["aspects"][aspect]["score"]
            score_list.append([aspect, company["aspects"][aspect]["score"]])
            count += 1
            max_score += 10
        avg_score = total_score / count
        table += f"| **Total Score** | {total_score}/{max_score} | |\n"
        table += f"| **Average Score** | {avg_score:.2f}/10.00 | |\n"
        st.markdown(table)
        fig, ax = plt.subplots()
        plt.grid(True, axis='x')
        score_df = pd.DataFrame(score_list, columns=["aspect", "score"])
        score_df = score_df.sort_values(by="score")
        thresh_select = st.radio("Select Threshold for Color Coding", ["Company Average", "Average Across All Companies"], key=company)
        thresh = avg_score if thresh_select == "Company Average" else global_avg_score
        ax.barh(score_df[score_df["score"] < thresh]["aspect"], score_df[score_df["score"] < thresh]["score"], color="red", zorder=2, label="Negative")
        ax.barh(score_df[score_df["score"] == thresh]["aspect"], score_df[score_df["score"] == thresh]["score"], color="cornflowerblue", zorder=2, label="Neutral")
        ax.barh(score_df[score_df["score"] > thresh]["aspect"], score_df[score_df["score"] > thresh]["score"], color="green", zorder=2, label="Positive")
        plt.axvline(avg_score, linewidth=1, color="k", linestyle="dashed", label="Company Average")
        plt.axvline(global_avg_score, linewidth=1, color="k", label="Average Across All Companies", zorder=3)
        plt.xlim(0, 10)
        plt.xticks(np.arange(0.0, 11.0, 1.0), zorder=0)
        ax.legend(bbox_to_anchor=(0, 1), loc='lower left', fontsize='small')
        st.pyplot(fig)
        st.write(f"## Competitor Comparison: \n {company["competitor_comparison_descriptive"]}")
        st.divider()

def individual_setfit_prediction(model, df):
    preds = []
    cleaned_df = df.iloc[:, 0].dropna()
    for cell in cleaned_df:
        preds.append(model.predict(cell))
    return preds

def extract_aspect_frequency(preds):
    aspect_dict = {}

    for i in preds:
        for j in i:
            if len(j) == 0:
                break
            if j["span"] not in aspect_dict:
                aspect_dict[j["span"]] = {"positive": 0, "neutral": 0, "negative": 0}

            if j["polarity"] == "positive":
                aspect_dict[j["span"]]["positive"] += 1
            elif j["polarity"] == "neutral":
                aspect_dict[j["span"]]["neutral"] += 1
            else:
                aspect_dict[j["span"]]["negative"] += 1

    aspect_df = pd.DataFrame.from_dict(aspect_dict)
    aspect_df = aspect_df.T
    return aspect_df

def build_pie_chart(aspect_df):
    fig, ax = plt.subplots()
    aspect_sum_df = aspect_df.sum()
    ax.pie(aspect_sum_df, labels=None, autopct="%1.1f%%", colors=["Green", "cornflowerblue", "Red"])
    ax.legend(labels=aspect_sum_df.index, bbox_to_anchor=(0, 1), loc='lower left', fontsize='small')
    return fig, ax

def build_bar_chart(aspect_df):
    fig, ax = plt.subplots()
    aspect_sum_df = aspect_df.sum()
    ax.bar(["Positive"], aspect_sum_df[0], color="g")
    ax.bar(["Neutral"], aspect_sum_df[1], color="cornflowerblue")
    ax.bar(["Negative"], aspect_sum_df[2], color="r")
    return fig, ax


def display_aspects(df):
    col1, col2 = st.columns(2)
    col1.dataframe(df)
    col1.dataframe(df.sum())
    fig, ax = build_pie_chart(df)
    col2.pyplot(fig)
    fig, ax = build_bar_chart(df)
    col2.pyplot(fig)

def less_detail(file1, file2, file3):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)
    df_combined = pd.DataFrame(np.concatenate([df1, df2, df3], axis=0), columns=df1.columns)
    with st.spinner("Analyzing..."):
        preds_1 = individual_setfit_prediction(model, df1)
        preds_2 = individual_setfit_prediction(model, df2)
        preds_3 = individual_setfit_prediction(model, df3)
        preds_combined = individual_setfit_prediction(model, df_combined)
        aspect_df_1 = extract_aspect_frequency(preds_1)
        aspect_df_2 = extract_aspect_frequency(preds_2)
        aspect_df_3 = extract_aspect_frequency(preds_3)
        aspect_df_combined = extract_aspect_frequency(preds_combined)
    st.write("## Frequency of Aspects per Company")
    st.write("### Company A")
    display_aspects(aspect_df_1)
    st.divider()
    st.write("### Company B")
    display_aspects(aspect_df_2)
    st.divider()
    st.write("### Company C")
    display_aspects(aspect_df_3)
    st.divider()
    st.write("### All Companies")
    display_aspects(aspect_df_combined)

st.write("# Aspect-Based Sentiment Analysis for Product Reviews")
plt.rcParams['font.family'] = 'sans-serif'
model = load_setfit_model()
with st.form("upload_form"):
    st.write("Please upload 3 .csv or .txt files of all reviews with this format: ")
    csv_col, txt_col = st.columns(2)
    csv_col.write("""
                ### CSV Format:
                A single column CSV containing one review enclosed in quotations per row.
                """)
    csv_col.dataframe(pd.DataFrame([["this is a review."], ["this is another review."], ["this is the third review."]]))
    txt_col.write("""
                ### TXT Format:
                A text file with one review enclosed in quotations per line.
                """)
    txt_col.code("""
                    \"this is a review.\"
                    \"this is another review.\"
                    \"this is the third review.\"
                """, language=None)
    detail = st.radio("Which Model?", ["SetFit", "Phi-3 Mini **GPU Unavailable**"])
    file1 = st.file_uploader("Company A Reviews", key="file1", type=[".csv", ".txt"])
    file2 = st.file_uploader("Company B Reviews", key="file2", type=[".csv", ".txt"])
    file3 = st.file_uploader("Company C Reviews", key="file3", type=[".csv", ".txt"])
    st.form_submit_button("Analyze")
detail = False if detail == "SetFit" else True
if (file1 and file2 and file3) or detail:
    if detail:
        more_detail()
    else:
        less_detail(file1, file2, file3)
else:
    st.warning("Please upload 3 files")