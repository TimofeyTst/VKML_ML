import numpy as np
from multiprocessing import Pool
from functools import partial
from transformers import DistilBertTokenizerFast, DistilBertModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

# Загрузка данных
train_data = pd.read_csv('train.csv')

# Инициализация токенизатора и модели
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def process_text(text, tokenizer, model):
    print("process_text begin")
    inputs = tokenizer(text, return_tensors='pt', truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    print("process_text end")
    return embeddings.detach().numpy()

def parallel_process_text(texts, tokenizer, model, num_workers=4):
    print("parallel_process_text begin")
    with Pool(num_workers) as pool:
        func = partial(process_text, tokenizer=tokenizer, model=model)
        results = pool.map(func, texts)
    print("parallel_process_text end")
    return np.array(results)

num_workers = 4
chunk_size = 1000

print("Before main for loop")
X = []
for i in range(0, len(train_data['title']), chunk_size):
    texts = train_data['title'][i:i+chunk_size]
    embeddings = parallel_process_text(texts, tokenizer, model, num_workers)
    X.append(embeddings)
    print(f"Processed chunk {i//chunk_size+1} / {len(train_data['title'])//chunk_size+1}")
print("After main for loop")

X = np.vstack(X)
X = pd.DataFrame(X)


with open('embeddings.pkl', 'wb') as f:
    pickle.dump(X, f)