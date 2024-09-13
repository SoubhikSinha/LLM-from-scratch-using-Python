# Large Language Model (LLM) from scratch using Python

### Acknowledgements
I would like to extend my sincere thanks to **[freeCodeCamp](https://www.freecodecamp.org/)** and the creator of the **[tutorial](https://www.youtube.com/watch?si=bKUnRhoGHbYjT5Ay&v=UU1WVnMk4E8&feature=youtu.be)** for their invaluable content and guidance in helping me build this project. This project wouldn't have been possible without their educational resources.

---

<br>
<br>

## Introduction : What is an LLM? 🤔

Imagine you’re texting a friend, and they ask :<br>

**"Hey, do you know how to bake banana bread?"**<br>

You pause... baking isn't exactly your strong suit. But instead of admitting defeat, you turn to your trusty LLM (Large Language Model), which instantly gives you the perfect banana bread recipe — with tips on how to make it extra fluffy. 🍞🍌
<br>
<br>

Crisis averted! Thanks to the LLM, you’re now the banana bread expert your friend thinks you are.

---

<br>

Well, that's a small funny introduction of one of the use cases of LLMs. Now, let's dive into the technical world of LLMs !
<br>

Large Language Models are the backbone of many advanced AI systems today. They are trained on vast amounts of text data, enabling them to understand, generate, and even engage in meaningful conversations with users. In this project, we'll explore how LLMs work from the ground up, covering everything from tokenization and embeddings to training techniques and model fine-tuning.

<br>

According to **AWS**🔻
> **Large Language Models (LLMs)** are advanced deep learning models pre-trained on massive datasets, typically utilizing the transformer architecture. Transformers consist of an encoder and decoder with self-attention mechanisms, enabling them to understand relationships between words and extract meanings from text sequences. Unlike earlier RNNs that process inputs sequentially, transformers handle entire sequences in parallel, making them more efficient and faster to train using GPUs. Their architecture supports massive models, often with hundreds of billions of parameters, allowing them to ingest vast amounts of data from sources like Common Crawl and Wikipedia, learning grammar, language, and knowledge through self-learning.

<br>

Now that you have an introductory knowledge about **LLMs**, it's time to dive into the components of this repo 🔻<br>

> **NOTE** : Because the notebook size are pretty big - you may not be able to render the notebook / code blocks. Thus, it is better to ***fork*** the repository / download the individual files to go through them.

<br>
<br>

### [PyTorch Basic Functions](https://github.com/sricks404/LLM-from-scratch-using-Python/blob/main/pytorch_basic_funcs.ipynb) 👇 <br>

This notebook contains all the necessary basic functions (from PyTorch) needed to build the LLM.

<br>
<br>

### [PyTorch CUDA GPU](https://github.com/sricks404/LLM-from-scratch-using-Python/blob/main/pytorch_CUDA_GPU.ipynb) 👇 <br>

Since the final LLM model will be trained on the **[OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext)** dataset, which was also used to train the **GPT-2** model, training on a CPU would take an extremely long time ☠️. Instead of using a CPU, we trained the model on a GPU (Graphical Processing Unit). The GPU used here is the **NVIDIA GeForce GTX 1050**.<br><br>

Incorporating GPU acceleration (NVIDIA CUDA) into model training can be challenging. This notebook gives you an idea how fast GPUs are as compared to the CPUs.

<br>

> **NOTE**: Each notebook in this repository is thoroughly commented, ensuring that you understand the "_Why's and How's_". The comments will provide a clear explanation as you go through the code.

<br>
<br>

### [Bi-Gram](https://github.com/sricks404/LLM-from-scratch-using-Python/blob/main/bigram.ipynb) 👇 <br>

A **`Bi-Gram Model`** is a straightforward yet powerful tool in Natural Language Processing (NLP) and language modeling. It works by looking at pairs of consecutive words in a text to predict what comes next. For example, if we consider the sentence "I love cats," the bi-grams would be "I love" and "love cats." By examining how often these word pairs appear together, the model learns the likelihood of a word following another. This helps in generating text that feels natural. However, while bi-gram models capture basic word relationships, they only consider immediate word pairs, so they might miss out on more complex or longer-term patterns in language. Despite this, they’re a key building block for understanding how words fit together and are often used as a foundation for more advanced models.
<br>
<br>
*Keywords : Batch Size, Block Size, Encoder-Decoder, Tensors, Optimizers*
<br>
<br>
This **Bi-Gram Model** is trained on the dataset (or rather, a book) sourced from **Project Gutenberg: [The Adventures of Sherlock Holmes](https://www.gutenberg.org/ebooks/1661)**

> **NOTE**: The focus of this project was on building the model, not on data preprocessing, so that step has been omitted.

<br>
<br>

### [GPT - Language Model (Version #1)](https://github.com/sricks404/LLM-from-scratch-using-Python/blob/main/gpt-v1%20-%201.ipynb) 👇<br>

Now that you have explored the implementation of the **Bi-Gram** model, we can take the first step in building our initial LLM model: _**GPT-v1 (Version 1)**_.

> **NOTE**  : This GPT model is also trained on the book: [The Adventures of Sherlock Holmes](https://www.gutenberg.org/ebooks/1661). In a subsequent notebook, you will see the model trained on the **OpenWebText** dataset.

<br>
<br>

# ⚠️ README.md under construction
