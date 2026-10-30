import sys
import csv
import re
from collections import defaultdict
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def load_templates():
    templates = defaultdict(list)
    try:
        with open('trainset.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                mr = row['mr']
                ref = row['ref']
                # Parse MR attributes (ignore familyFriendly)
                attr_dict = {}
                for attr, value in re.findall(r'(\w+)\[([^\]]+)\]', mr):
                    if attr != 'familyFriendly':
                        attr_dict[attr] = value
                # Delexicalize reference
                delex_text = ref
                for attr, value in sorted(attr_dict.items(), key=lambda x: len(x[1]), reverse=True):
                    delex_text = re.sub(rf'\b{re.escape(value)}\b', attr.upper(), delex_text, flags=re.IGNORECASE)
                # Store template by attribute set
                attr_set = frozenset(attr_dict.keys())
                templates[attr_set].append(delex_text)
    except FileNotFoundError:
        print("Error: trainset.csv not found. Please ensure it is in the same directory.")
        sys.exit(1)
    
    # Select most frequent template for each attribute set
    return {attr_set: max(template_list, key=template_list.count) for attr_set, template_list in templates.items()}

def generate_text(mr, templates):
    attr_dict = {}
    for attr, value in re.findall(r'(\w+)\[([^\]]+)\]', mr):
        if attr != 'familyFriendly':
            attr_dict[attr] = value
    attr_set = frozenset(attr_dict.keys())
    
    #Try exact match
    if attr_set in templates:
        text = templates[attr_set]
        for attr, value in attr_dict.items():
            text = text.replace(attr.upper(), value)
        return text
    
    # If there is a Partial match: find largest valid subset
    best_attrs = max((s for s in templates if s.issubset(attr_set)), key=len, default=None)
    if best_attrs:
        text = templates[best_attrs]
        partial_attrs = {k: v for k, v in attr_dict.items() if k in best_attrs}
        for attr, value in partial_attrs.items():
            text = text.replace(attr.upper(), value)
        return text
    
    #Fallback if no templates match
    return attr_dict.get('name', 'A restaurant') + ' is a restaurant.'

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python templatesys.py testset_w_refs.csv")
        sys.exit(1)
    
    test_file = sys.argv[1]
    
    # Load templates from training file
    TEMPLATES = load_templates()
    
    references = []
    hypotheses = []
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Generate text for each MR in the test set
                hypotheses.append(generate_text(row['mr'], TEMPLATES).split())
                references.append([row['ref'].split()])
    except FileNotFoundError:
        print(f"Error: {test_file} not found.")
        sys.exit(1)
    
    # Calculate and print the BLEU score
    bleu_score = corpus_bleu(references, hypotheses, smoothing_function=SmoothingFunction().method4)
    print(f"{bleu_score:.4f}")