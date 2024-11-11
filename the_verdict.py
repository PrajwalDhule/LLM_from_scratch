import urllib.request
import re

# reading the text file from the URL of the text file
url = ("https://raw.githubusercontent.com/rasbt/"
      "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
      "the-verdict.txt")
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
   raw_text = f.read()
print("Total number of character:", len(raw_text))
print(raw_text[:99])

# Preprocessing the text to split the text into tokens
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
print(preprocessed[:30])


# Building a vocabulary from the preprocessed (tokenised) text
all_words = sorted(set(preprocessed))
all_words.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_words)}

class SimpleTokenizerV2:    
      def __init__(self, vocab): 
            self.str_to_int = vocab           
            self.int_to_str = {i:s for s,i in vocab.items()}           
      
      def encode(self, text):        
            preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text) 
            preprocessed = [item.strip() for item in preprocessed if item.strip()]
            preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
             
            ids = [self.str_to_int[s] for s in preprocessed] 
            return ids    
      
      def decode(self, ids):        
            text = " ".join([self.int_to_str[i] for i in ids]) 
            text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)   
            return text
      
# tokenizer = SimpleTokenizerV1(vocab) 
# text =  """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride.""" 
# ids = tokenizer.encode(text) 
# print(tokenizer.decode(ids))

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text3 = "painted."
text4 = "pardonable."
text5 = "lmao."
text = " <|endoftext|> ".join((text1, text2, text3, text4, text5))
print(text)
tokenizer = SimpleTokenizerV2(vocab)
# print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))

