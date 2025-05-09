# 🧠 Natural Language Classification: Quora Insincere Questions

## 📌 Project Overview
This project aims to classify questions from the Quora Insincere Questions Classification challenge using the Bag of Words (BoW) technique and traditional machine learning methods.

The goal is to detect toxic or insincere questions that could undermine constructive discourse on the platform.

## 🗂️ Project Structure
The project is divided into the following main steps:

- Download and Explore the Data

- Text Preprocessing

- Feature Engineering using Bag of Words

- Model Training and Evaluation

- Prediction and Submission to Kaggle

### 📥 1. Data Download and Exploration

Dataset downloaded using the Kaggle API and stored in a data/ directory.

Files used:

- train.csv – training data

- test.csv – test data

- sample_submission.csv – sample submission format

#### Explore the Data Using Pandas

train_fname = 'data/train.csv.zip'
raw_df = pd.read_csv(train_fname)
raw_df

   qid                                  	question_text	                 target
0	00002165364db923c7e6	How did Quebec nationalists see their province...	0
1	000032939017120e6e44	Do you have an adopted dog, how would you enco...	0
2	0000412ca6e4628ce2cf	Why does velocity affect time? Does velocity a...	0
3	000042bf85aa498cd78e	How did Otto von Guericke used the Magdeburg h...	0
4	0000455dfa3e01eae3af	Can I convert montra helicon D to a mountain b...	0
...	...	...	...
1306117	ffffcc4e2331aaf1e41e	What other technical skills do you need as a c...	0
1306118	ffffd431801e5a2f4861	Does MS in ECE have good job prospects in USA ...	0
1306119	ffffd48fb36b63db010c	Is foam insulation toxic?	0
1306120	ffffec519fa37cf60c78	How can one start a research project based on ...	0
1306121	ffffed09fedb5088744a	Who wins in a battle between a Wolverine and a...	0
1306122 rows × 3 columns

# Create the bar plot object
ax = raw_df.target.value_counts(normalize=True).plot(
    kind='bar',
    color=['skyblue', 'salmon'],  # colors for bars
    figsize=(8, 6)
)

# Add title and axis labels
ax.set_title('Distribution of Sincere and Insincere Questions in the training data', fontsize=14)
ax.set_xlabel('Question Type (0 = Sincere, 1 = Insincere)', fontsize=12)
ax.set_ylabel('Proportion', fontsize=12)

# Show percentage values on top of bars
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2%}', (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

![image](https://github.com/user-attachments/assets/e26bd9d2-b3bc-4d1a-95a9-f22b15557e3f)


### 🧹 Apply Text Preprocessing Techniques

Outline:
- Understanding the Bag of Words Model
- Tokenization
- Stop Words removal
- Stemming

#### Bag of words intuition

- Create a list of all the words across all the text document.

- Convert each document (Question) in to a vector containing the counts of each word.

#### Limitations

- There may be too many words in the dataset - The vector may not fit in the memory

- Some words may occure too frequentlly

- Some words may occure vey rarely or only once - It may not give enough information the calassify the word appropriately

A single word may have many forms (go,gone,going bird vs birds)

### 🧮 Implement the Bag of Words Model

Outline:

- Create a vocabulary using Count Vectorizer

- Transform text to Vectors using Count Vectorizer

- Configure text Preprocessing in Count Vectorizer

### 🤖 Train ML Model for Classification

- Create a training and validation set
  
- Train a logistic regression model
  
- Make predictions on training, validation and test data.

### 📤 Make Predictions and Submit to Kaggle

