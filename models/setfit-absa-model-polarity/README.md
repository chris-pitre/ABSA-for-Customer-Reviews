---
base_model: sentence-transformers/paraphrase-mpnet-base-v2
library_name: setfit
metrics:
- accuracy
pipeline_tag: text-classification
tags:
- setfit
- absa
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: have a good meal at a fair:Tuk Tuk is one of those comfortable neighborhood
    joints where you know you will always have a good meal at a fair price.
- text: good selection of wines ranging from affordable:good selection of wines ranging
    from affordable to high end.
- text: relax and the food is one of:Its an excellent place to relax and the food
    is one of the best in the city of New York.
- text: awful - the sugar was charred,:Food was average and creme brulee was awful
    - the sugar was charred, not caramelized and smelled of kerosene.
- text: service is friendly,:service is friendly, and never had a problem walking
    in and getting a table.
inference: false
---

# SetFit Polarity Model with sentence-transformers/paraphrase-mpnet-base-v2

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Aspect Based Sentiment Analysis (ABSA). This SetFit model uses [sentence-transformers/paraphrase-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2) as the Sentence Transformer embedding model. A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification. In particular, this model is in charge of classifying aspect polarities.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

This model was trained within the context of a larger system for ABSA, which looks like so:

1. Use a spaCy model to select possible aspect span candidates.
2. Use a SetFit model to filter these possible aspect span candidates.
3. **Use this SetFit model to classify the filtered aspect span candidates.**

## Model Details

### Model Description
- **Model Type:** SetFit
- **Sentence Transformer body:** [sentence-transformers/paraphrase-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2)
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **spaCy Model:** en_core_web_lg
- **SetFitABSA Aspect Model:** [/content/drive/MyDrive/ABSA/models/setfit-absa-model-aspect](https://huggingface.co//content/drive/MyDrive/ABSA/models/setfit-absa-model-aspect)
- **SetFitABSA Polarity Model:** [/content/drive/MyDrive/ABSA/models/setfit-absa-model-polarity](https://huggingface.co//content/drive/MyDrive/ABSA/models/setfit-absa-model-polarity)
- **Maximum Sequence Length:** 512 tokens
- **Number of Classes:** 4 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label    | Examples                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
|:---------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| negative | <ul><li>'But the staff was so horrible:But the staff was so horrible to us.'</li><li>', forgot our toast, left out:They did not have mayonnaise, forgot our toast, left out ingredients (ie cheese in an omelet), below hot temperatures and the bacon was so over cooked it crumbled on the plate when you touched it.'</li><li>'did not have mayonnaise, forgot our:They did not have mayonnaise, forgot our toast, left out ingredients (ie cheese in an omelet), below hot temperatures and the bacon was so over cooked it crumbled on the plate when you touched it.'</li></ul>                                          |
| positive | <ul><li>"factor was the food, which was:To be completely fair, the only redeeming factor was the food, which was above average, but couldn't make up for all the other deficiencies of Teodora."</li><li>"The food is uniformly exceptional:The food is uniformly exceptional, with a very capable kitchen which will proudly whip up whatever you feel like eating, whether it's on the menu or not."</li><li>"a very capable kitchen which will proudly:The food is uniformly exceptional, with a very capable kitchen which will proudly whip up whatever you feel like eating, whether it's on the menu or not."</li></ul> |
| neutral  | <ul><li>"'s on the menu or not.:The food is uniformly exceptional, with a very capable kitchen which will proudly whip up whatever you feel like eating, whether it's on the menu or not."</li><li>'to sample both meats).:Our agreed favorite is the orrechiete with sausage and chicken (usually the waiters are kind enough to split the dish in half so you get to sample both meats).'</li><li>'to split the dish in half so:Our agreed favorite is the orrechiete with sausage and chicken (usually the waiters are kind enough to split the dish in half so you get to sample both meats).'</li></ul>                   |
| conflict | <ul><li>'The food was delicious but:The food was delicious but do not come here on a empty stomach.'</li><li>"The service varys from day:The service varys from day to day- sometimes they're very nice, and sometimes not."</li></ul>                                                                                                                                                                                                                                                                                                                                                                                         |

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import AbsaModel

# Download from the 🤗 Hub
model = AbsaModel.from_pretrained(
    "/content/drive/MyDrive/ABSA/models/setfit-absa-model-aspect",
    "/content/drive/MyDrive/ABSA/models/setfit-absa-model-polarity",
)
# Run inference
preds = model("The food was great, but the venue is just way too busy.")
```

<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Set Metrics
| Training set | Min | Median  | Max |
|:-------------|:----|:--------|:----|
| Word count   | 6   | 21.3594 | 43  |

| Label    | Training Sample Count |
|:---------|:----------------------|
| conflict | 2                     |
| negative | 19                    |
| neutral  | 25                    |
| positive | 82                    |

### Training Hyperparameters
- batch_size: (32, 32)
- num_epochs: (5, 5)
- max_steps: -1
- sampling_strategy: oversampling
- body_learning_rate: (2e-05, 1e-05)
- head_learning_rate: 0.01
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: True
- warmup_proportion: 0.1
- seed: 42
- eval_max_steps: -1
- load_best_model_at_end: True

### Training Results
| Epoch      | Step    | Training Loss | Validation Loss |
|:----------:|:-------:|:-------------:|:---------------:|
| 0.0037     | 1       | 0.2612        | -               |
| 0.1845     | 50      | 0.1948        | 0.2081          |
| **0.3690** | **100** | **0.0118**    | **0.1986**      |
| 0.5535     | 150     | 0.0013        | 0.2254          |
| 0.7380     | 200     | 0.0007        | 0.235           |
| 0.9225     | 250     | 0.0003        | 0.2263          |
| 1.1070     | 300     | 0.0002        | 0.231           |
| 1.2915     | 350     | 0.0004        | 0.2538          |

* The bold row denotes the saved checkpoint.
### Framework Versions
- Python: 3.10.12
- SetFit: 1.0.3
- Sentence Transformers: 3.0.1
- spaCy: 3.7.5
- Transformers: 4.40.2
- PyTorch: 2.3.0+cu121
- Datasets: 2.20.0
- Tokenizers: 0.19.1

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->