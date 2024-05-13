from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import pandas as pd



def paraphrase_text(text, mode='elaborate', num_beams=2, max_length=100, min_length=30):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs['input_ids'],
        num_beams=num_beams,
        max_length=max_length,
        min_length=min_length,
        length_penalty=1.0 if mode == 'summarize' else 2.0,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

if __name__=='__main__':
  # Load model and tokenizer
    model_name = 'facebook/bart-base'
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    # Example usage:
    data=pd.read_csv('nlu_labeled.csv')
    sample_elaboration = data.loc[0, 'elaboration']
    sample_summary = data.loc[0, 'summary']

    paraphrased_elaboration = paraphrase_text(sample_elaboration, mode='elaborate')
    paraphrased_summary = paraphrase_text(sample_summary, mode='summarize')

    paraphrased_elaboration, paraphrased_summary
