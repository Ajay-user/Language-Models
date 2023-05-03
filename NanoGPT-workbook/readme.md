## Modeling Natural Language

#### Word2vec --> map words to vectors

<img src='./Notes/1 vectors in ndim space.webp'>

<br>
<br>

When modeling natural language, we treat words as continuous vectors in the N-dimensional space.

**Learning word vectors**

We can learn word vectors in two ways

- Supervised learning
  - → learn word vectors by training on a supervised task
  - → Here we need labeled data and getting data labeled is both time intensive and cost intensive.

<br>

- Unsupervised learning
  - → We like to learn word vectors from the text corpus itself instead of requiring a human to label them first and then learning the word vectors by training on a supervised task.

<br>
<br>

### Good word vectors are word vectors for which using those vectors we’re capable of predicting the words that might be surrounding a given word.

<img src='./Notes/0 word to vec.webp'>

<br>
<br>

#### We want to map each word to a vector in the n-dim space

<h1 style='color:lightblue'> " A little red fluffy <span style='color:red'>dog</span> wanders down the road "</h1>

<br>

<img src='./Notes/1.1 cbow.webp'>

- In CBOW architecture we use the context around the words as input to the model.
- We take the word vectors for all the words in the neighborhood of the nth-word “ dog ” .
- Then we average the word vectors to get a fixed sized input. **This process in called as bagging.**
  <br>

#### Draw backs of this approach :

- Continuous bag of word model doesn’t take into account the word order

  - **Here is an another example** : The united states of `______`

- What could be the next possible word ?

  - **Consider this sequence** : The states united of `______`

**another example**

- `"a dog is chasing a man"` ---- `"a man is chasing a dog"`

<br>

<img src='./Notes/1.2 skipgram.webp'>

#### In Skip Gram architecture instead of using the context words as input, we use the nth word as input and ask the model to predict the surrounding words in a sequence.

<br>

<img src='./Notes/1.3 cost function.webp'>

- If we can optimize this cost-function then it means we have a model that is good at predicting the desired output given the input.
- Through this process of optimization we have learned good embedding vectors for the words in our vocabulary

<br>

## Humans build representations overtime. We use these representation to predict what comes next.

<h4 style='color:salmon'>— “john loves dogs and his dogs wags their <span style='color:seagreen'>???????????” —</span></h4>

<h4> Guess the next word  </h4>

The fact that you were able to guess the next word suggest that you build up a representation and you use that representation to predict the next word.

<br>

**The idea behind word vectors**

- Every language has many rules inplace when constructing a sentence.
- The words in a sentence follow these rules.
- This implies that for each word in a given sentence we should be able to predict the presence of surrounding words.

## Why is context important ?

This concept of mapping each word in a document to a single vector is restrictive, in the sense that words has multiple meaning and the meaning that applies corresponds to the context in which the word is used.

For example : Consider the following sentences

<h5>“The bank <span style='color:salmon'>appropriated</span> the property of the defaulters”</h5>
<h5>“There can be problems in  <span style='color:salmon'>appropriating</span> funds for legal expenses”</h5>
<h5>"Teaching  <span style='color:salmon'>appropriate</span>  Behavior in the work place”</h5>

<br>
<br>
<br>

- By Associating a word to a fixed embedding we cannot handle cases where words have multiple meanings.

**Suppose we are encoding the word “ `bank` ”.**

The context independent encoding will map the word “ `bank` ” in the sentence `“On the river bank”` and `“Open an account in the bank”` to the same embedding vector.

Ideally we want the word embedding to be more contextualized to reflect the surrounding words around them.

So that the word “ `bank` ” in the sentence `“On the river bank”` will be mapped different from the word “ `bank` ” in the sentence word `“Open an account in the bank”`

<img src='./Notes/1.4 context is important.webp'>

<br>
<br>
<br>

**The underlying components of the word vector have some thematic meaning.**

<img src='./Notes/2 components of word vector.webp'>

<br>

### If we can learn these word vectors by taking the context into account

<img src='./Notes/3 properties of word embedding.webp'>

<br>

#### For example words like ‘king’ and ‘queen’. By taking these word vectors, we can look for what is similar between them, what is different, etc… and from that we can uncover some underlying information associated with these word vectors.

<img src='./Notes/4 topics in components.webp'>

<br>

#### What we would like to do now is to develop a framework by which we can learn these word vectors in a way that takes into account the meaning implied by the surrounding words [ contextual information ] and the word order.

#### Meaning

- Similar words should be close in the embedding space

#### Context

- Word's have different meaning depending on the context in which its used

#### Word order

- Word order is important because there are rules that dictate how words unfold in a sentence
- `"a dog is chasing a man"` ---- `"a man is chasing a dog"`
- When word order changes, meaning of the sentence changes

<br>

## ATTENTION IS ALL YOU NEEDED

- **Quantifying the similarity between words**
  - We can use dot product (inner product) to quantify the similarity between words
  - The components of word vectors are composed of different themes.
  - If the word aligns with the theme then the corresponding component will be a positive number.
  - If the word doesn’t aligns with the theme then the corresponding component will be a negative number.
  - The dot product between two word vectors give us similarity between two words

<img src='./Notes/5 dotproduct for similarity.webp'>
<br>
<br>

- **Relative similarity score**

  - Relative similarity R(k →i) implies how similar is the kth word to the ith word, relative to all other words in the sequence.
  - **Quantifying how similar is the `word(i)` to `word(k)`**
    <br>
    <br>
    <img src='./Notes/5.1 softmax.webp'>

#### The relative similarity score is a mathematical way of quantifying the relationship between words and those relationships between words are characteristic of language itself. These relationship dictates how words unfold in a sentence.

<img src='./Notes/6 attention mechanism.webp'>

#### The word vectors we started with was not informed of the context in which it was used. By computing the relative similarity score, we now have the information about how much attention we should pay to the surrounding words.

## Self ATTENTION | context awareness

#### The words have different meaning depending on the context in which its used. So mapping word to a single vector is restrictive in the sense that it implies only one kind of meaning

- Example : “On the river bank they sit” , “The bank control the interest rate”

<img src='./Notes/7 attention block diagram.webp'>

<h2 style='color:lightblue'>
Self attention helps encoding word to vectors by taking contextual information into account</h2>
<h2 style='color:salmon'>
Self attention does not takes into account word order
</h2>

**When word order changes, the meaning changes. The attention mechanism will give us word vectors they are independent of word order. Changing the word order doesn’t change the word encoding produced by the attention mechanism.**

## Positional Embedding

The word vectors are d-dimensional vectors. We want to incorporate the positional information into these word vectors. The positional embedding are d-dimensional vectors that reflect the position of the word in a sequence. For every component in this vector we are going to associate a sine/cosine wave of different frequency.

$$Positional \: Embedding$$

$$ PE*{(position, 2i)} = \sin({\frac{positions}{10000^{\frac{2i}{dims}}}})$$
$$ PE*{(position, 2i+1)} = \cos({\frac{positions}{10000^{\frac{2i}{dims}}}})$$

<img src='./Notes/7.1 positional embeddings.webp'>

## Skip connection

We want to preserve the meaning of the word, the word embedding and account for the contextual information. When we go from input at the bottom to output at the top, we are losing the original word embedding. The skip connection help us preserve the original embedding.

Skip connection also help us prevent vanishing gradient problem

## Improved ATTENTION

<img src='./Notes/8 meaning word-order and context.webp'>

<br>

## Deep Sequence Encoder

#### The sequence encoder is a stack of different layers. In the bottom we have word embedding layer and positional embedding. The next set of layers are Attention network and FFNN with skip connection in between. This set of layers can be repeated ‘K’ times to form a deep sequence encoder, which will improve the performance of the network.

<img src='./Notes/9 Deep sequence encoder.webp'>
