# filters.py

non_clinical_terms = [
    "inner child", "trauma bonding", "energy work", 
    "attachment style", "chakra", "crystals", "spiritual awakening"
]

def detect_non_clinical_terms(text):
    flags = []
    for term in non_clinical_terms:
        if term.lower() in text.lower():
            flags.append(term)
    return flags
