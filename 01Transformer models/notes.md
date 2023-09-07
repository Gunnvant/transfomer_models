The course is basically about using hugging-face ecosystem to do nlp with transformers.

## Introduction

Three sections planned:

1. Introduction (Current course)
- Transformer models
- Using hugging face transformers
- Fine tuning a pre-trained model
- Sharing models and tokenizers

2. Diving in (Planned fall 2021)
- The datasets library
- The tokenizers library
- Main NLP tasks
- How to ask for help

3. Advanced (End 2021 or early 2022)
- Specialized architectures
- Speeding up training
- A custom training loop
- Contributing to hugging face

## NLP

The way hugging face folks have categorized the field of NLP is by looking at the tasks commonly tackled (my sense is this is not the complete list of nlp tasks, just nlp done by transformer models in the contemporary sense)

- Classifying whole sentences: For example, getting the sentiment of a sentence, finding out if a sentence is grammatically correct or not etc
- Classifying each work in the text: For example doing POS tagging, NER etc.
- Generating text: For example building auto-complete systems. 
- Question/Answering: Given a question and a context, extract the answer
- Generating new sentence from text: Translating text, summarizing text

## Pipelines in Hugging face

One of the simplest ways to get started with hugging face library is to use pre-trained models with `pipelines`. Pipelines implements model inference along with pre-processing steps.

Following pipelines are available:

- feature-extraction (extract vector representation of text)
- fill-mask (autocomplete system?)
- ner (classifying each word in a text)
- question-answering (question-answering)
- sentiment-analysis (classifying whole sentences)
- summarization (generating new sentences from text)
- text-generation (generating text)
- translation (generating new sentences from text)
- zero-shot-classification (no counterpart discussed above)

**Zero Shot Classification:** This is interesting, helps one assign labels to a text that has not been labelled. Imagine you have tweets which are not labelled but you know that tweets can be either about technology, startups or sports. In a zero-shot classification scenario you can let the system choose one label out of the n-labels you've provided.

The api to work with pipelines is very straight-forward

```python
from transformers import pipeline

new_pipeline = pipeline("text-generation", model = "distillgpt2") 

## pipeline(name_pipeline,model=name of specific model you want to use)

```

## How transformers work:

Only two key takeaways :

1. Pre-training in the context of nlp is:
    - Building language models that predict the next word
    - One way to build language models is to use masks (BERT does that)
2. Read the paper "All you need is attention"


## Encoders

This course only gives a superficial understanding of encoders with self attention mechanism. Forward pass is not shown, positional encoding etc is not shown. Idea of what multi-head attention is not explained.

What this section talks about is interestingly the idea of where encoders can be used. BERT is an encoder only model and is suited for the following tasks:

1. Sentence classification
2. Word classification (NER, POS tagging etc)
3. Extractive Question Answering

Apart from this, the following two qualifications are important

- The pre-training of the encoders revolves around being able to predict the masked words.
- The attention mechanism is bi-directional

BERT family of models are encoder based.

## Decoders

Following qualifying remarks are made in the course:

1. The pre-training involves predicting the next word of a sequence
2. The attention is unidirectional 

Are mostly used when:

1. Text generation is the intention.

GPT family of models are decoder based.

## Sequence to Sequence Models

Content is again silent on specifics and gives an overview of sequence to sequence models

Salient points made are:

1. Decoders can produce and output whose length is different from input given to encoders.
2. These models work well when doing language translation or text summarization.
3. Pre-training not very clearly explained.

Common Sequence to Sequence models are BART, T5, mBART.

