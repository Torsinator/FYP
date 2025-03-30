from transformers import BertTokenizer, BertModel
import torch

def encode_text(input_text):
    # Load the pretrained BERT tokenizer and model.
    # For efficiency, might move these loads outside the function if encoding many texts.
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    # Tokenize the input text, returning PyTorch tensors.
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

    # Disable gradient updates; we only need inference.
    with torch.no_grad():
        outputs = model(**inputs)

    # Use the pooled output, which corresponds to the [CLS] token embedding.
    # For bert-base-uncased, this will be a 768-dimensional vector.
    pooled_output = outputs.pooler_output
    return pooled_output

if __name__ == "__main__":
    user_input = input("Enter text to encode with BERT: ")
    embedding = encode_text(user_input)
    print("BERT embedding shape:", embedding.shape)
    print("Embedding vector:")
    print(embedding)
