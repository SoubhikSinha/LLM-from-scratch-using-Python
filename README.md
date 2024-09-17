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

In this notebook, even though the model is still based on the **Bi-Gram Model**, we can see that the hyperparameters used are now different from what we saw in the **Bi-Gram Model Implementation**. This is because we are trying to build a **Transformer**, which is the primary building block for GPT model(s). Let us list them all here🔻

 - **block_size**: The length of input sequences the model processes at once (e.g., 32 to 128 tokens).
    
-   **batch_size**: Number of samples processed before updating the model weights (e.g., 64 to 512).
    
-   **max_iters**: Total number of training iterations (e.g., 1000 to 5000).
    
-   **learning_rate**: The step size for updating model weights, with potential decay over time (e.g., 1e-5 to 1e-3).
    
-   **eval_iters**: Frequency of evaluating the model during training (e.g., every 50 to 500 iterations).
    
-   **n_embd**: Dimensionality of the embedding vectors (e.g., 128 to 512).
    
-   **n_layer**: Number of layers in the model architecture (e.g., 6 to 12).
    
-   **n_head**: Number of attention heads in each layer (e.g., 4 to 12).
    
-   **dropout**: Proportion of neurons randomly set to zero to prevent overfitting (e.g., 0.1 to 0.3).

<br>

> **NOTE**: Even though the dataset is relatively small (just a book) compared to what we will use in the next version of our GPT model, and the model is being trained on a GPU, the training and validation process may still take a significant amount of time. This is highly dependent on the values chosen for each hyperparameter. Therefore, be patient 🙂 (and not 🤪) — results will take time to show up ⌛.

<br>

The main goal of our GPT model (for now) is to achieve the lowest Validation Loss and generate content that appears relevant and meaningful. Don’t get disheartened if you encounter some "alien-like" language being generated 👽.

<br>
<br>

### [GPT - Language Model (Version #2)](https://github.com/sricks404/LLM-from-scratch-using-Python/blob/main/gpt-v1%20-%202.ipynb) 👇<br>

Alright folks, it's time to dive into some serious stuff. The dataset we'll be using is **[OpenWebtext](https://huggingface.co/datasets/Skylion007/openwebtext)**. This is the same dataset on which **[OpenAI's GPT-2](https://openai.com/index/better-language-models/)** was trained. 

<br>

> **NOTE:** Due to the enormous size of the dataset and limited computing capacity, we've selected appropriate and "healthy" hyperparameter values to avoid overloading your system. Using inappropriate values could result in 💣💥☠️💀👻. Please use the appropriate settings to ensure a smooth and efficient process !




# ⚠️ README.md under construction
