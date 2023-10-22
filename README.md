<h1 align="center">Jatikarok and BanglaPRCorpus</h1>
<p align="center">
  <b>Advancing Bangla Punctuation Restoration by a Monolingual Transformer-Based Method and a Large-Scale Corpus</b> </br> 
  [Accepted at EMNLP 2023 Workshop BLP, Paper — <a href="https://arxiv.org/" target="_blank">Link will be updated</a>]
</p>

# Jatikarok in a Nutshell
![bigpicture](https://github.com/mehedihasanbijoy/Jatikarok-and-BanglaPRCorpus/assets/58245357/9ddd5536-8a72-45c5-ae60-9873639d3fa5)

</br> </br> 

# BanglaPRCorpus Statistic
The Bangla punctuation restoration corpus, christened as BanglaPRcorpus, is constituted by 1.48 million source-target pairs. Within these pairs, the omission of punctuation from source sentences is conspicuous, while the target sentences epitomize the rectified versions where the supplementation of missing punctuation is executed. The process of correction entails the methodical removal of punctuation marks across the sentences, spanning a spectrum of quantities, ranging from 1 to 10, within each sentence. Moreover, it is of significance to underscore that the sentences within our corpus manifest a divergence in length, with the minimum sentence being characterized by a mere 2 words, the maximum sentence expanding to a substantial 127 words, and the average sentence length averaging at 12.9 words.

</br> </br> 


# Get Started
Clone the GitHub repository of the paper.
```
git clone https://github.com/mehedihasanbijoy/Jatikarok-and-BanglaPRCorpus.git
```
Alternatively, you can manually **download** and **extract** the GitHub repository of Jatikarok-and-BanglaPRCorpus.

</br> </br> 

# Environment Setup
Install the required packages.
```
conda env create -f requirements.yml
```
Afterward, activate the virtual environment and navigate to the paper directory.
- ``conda activate jatikarok``
- ``cd Jatikarok and BanglaPRCorpus``

</br> </br> 

# Download the BanglaPRCorpus
```
gdown https://drive.google.com/drive/folders/1V1OrkJ4okSgw5swmhrbXAZFqkDB8g7QX?usp=share_link -O ./BanglaPRCorpus/BanglaPRCorpus/ --folder
```
<p>
or manually <b>download</b> the folder from <a href="https://drive.google.com/drive/folders/1V1OrkJ4okSgw5swmhrbXAZFqkDB8g7QX?usp=share_link" target="_blank">here</a> and keep the extracted files into <b>./BanglaPRCorpus/BanglaPRCorpus/</b>
</p>

# Regenerate the BanglaPRCorpus
Go to `./BanglaPRCorpus` directory and follow the instructions.

</br> </br> 

# Train/Validate/Evaluate the Model
The experiments in this paper involves benchmarking three methods, namely Jatikarok, BanglaT5, and T5 Small, on three different corpora, including BanglaPRCorpus, ProthomAloBalanced, and BanglaOPUS.

### To train, validate, and evaluate Jatikarok on BanglaPRCorpus
```
python main.py --CORPUS_PATH "./BanglaPRCorpus/BanglaPRCorpus/corpus.csv" --KNOWLEDGE_PATH "./KnowledgeToBeTransferred/gecJatikarok.pth" --CHECKPOINT_PATH "./ModelCheckpoints/prJatikarok.pth" --MODEL_NAME "jatikarok" --BATCH_SIZE 16 --N_EPOCHS 50
```
</br> </br> 

# Benchmarking Bangla Punctuation Restoration Task
![image](https://github.com/mehedihasanbijoy/Jatikarok-and-BanglaPRCorpus/assets/58245357/b168f78f-9b39-483f-9961-4eeffdc56c06)

