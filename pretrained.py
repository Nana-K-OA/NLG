import sys
import csv
import torch
from transformers import pipeline, set_seed
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


# NB: System 3 – Pretrained Language Model

# I attempted to implement this system using Hugging Face pre-trained models.
# Originally, I tried BLOOM (300M and 560M variants), but my laptop froze
# because I do not have a GPU and limited RAM. I also tried running a smaller 
# GPT-2 model, but it still caused heavy slowdowns and was not practical 
# on my hardware. 

# As a compromise, I switched to the smallest available option 
# (`distilgpt2`) and reduced the workload by processing only a fraction 
# of the test set (5%). I didnt finish a complete run to see Bleu score

def create_prompt(mr):
    examples = [
        ("name[The Eagle], area[riverside], familyFriendly[yes]", 
         "The Eagle is a family-friendly pub located by the riverside."),
        ("name[Blue Spice], eatType[restaurant], food[English]", 
         "Blue Spice is a restaurant that serves English food."),
        ("name[The Mill], near[The Rice Boat]", 
         "The Mill is a coffee shop found near The Rice Boat.")
    ]
    
    prompt = "Generate restaurant descriptions:\n\n"
    for ex_mr, ex_text in examples:
        prompt += f"MR: {ex_mr}\nText: {ex_text}\n\n"
    
    prompt += f"MR: {mr}\nText:"
    return prompt

def main():
    if len(sys.argv) != 2:
        print("Usage: python pretrained.py testset_w_refs.csv")
        sys.exit(1)
    
    # Use the smallest available model for fastest inference
    print("Loading tiny model for fast inference...")
    generator = pipeline(
        'text-generation', 
        model='distilgpt2',  # Much smaller than GPT-2
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32  # Faster on GPU
    )
    set_seed(42)
    
    references = []
    hypotheses = []
    test_file = sys.argv[1]
    
    # Only process every 5th example to reduce computation (5% of test set)
    with open(test_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        sampled_rows = rows[::20]  # Process only 5% of test data
        
        print(f"Processing {len(sampled_rows)} examples (5% of test set)")
        
        for i, row in enumerate(sampled_rows):
            prompt = create_prompt(row['mr'])
            
            try:
                result = generator(
                    prompt, 
                    max_length=len(prompt.split()) + 30,  # Generate 30 more tokens
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=50256,
                    truncation=True  # Faster processing
                )
                
                generated = result[0]['generated_text'].replace(prompt, '').strip()
                generated = generated.split('\n')[0].strip()
                
            except Exception as e:
                print(f"Error generating: {e}")
                generated = "A restaurant."
            
            hypotheses.append(generated.split())
            references.append([row['ref'].split()])
            
            if i < 3:
                print(f"MR: {row['mr']}")
                print(f"Generated: {generated}")
                print(f"Reference: {row['ref']}")
                print("---")
    
    # Calculate BLEU score
    if hypotheses:
        bleu_score = corpus_bleu(references, hypotheses, smoothing_function=SmoothingFunction().method1)
        print(f"BLEU score: {bleu_score:.4f}")
        print(f"Based on {len(hypotheses)} test examples (5% of full test set)")
    else:
        print("No examples processed")

if __name__ == '__main__':
    main()