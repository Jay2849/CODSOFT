import string
from tqdm import tqdm

# captions.txt फाइल को लोड करने का फंक्शन
def load_doc(filename):
    with open(filename, 'r') as file:
        text = file.read()
    return text

# कैप्शन को साफ़ करने का फंक्शन
def clean_descriptions(descriptions):
    # पहली लाइन (header) को हटाना
    lines = descriptions.strip().split('\n')
    caption_lines = lines[1:]

    cleaned_descriptions = {}

    # tqdm का इस्तेमाल करके लूप पर प्रोग्रेस बार लगाना
    print("Cleaning captions... Please wait.")
    for line in tqdm(caption_lines):
        # इमेज का नाम और कैप्शन को अलग करना
        parts = line.split(',', 1)
        if len(parts) < 2:
            continue
        image_id, image_desc = parts[0], parts[1]

        # इमेज के नाम से .jpg हटाना
        image_id = image_id.split('.')[0]

        # सारे punctuation हटाना
        image_desc = image_desc.translate(str.maketrans('', '', string.punctuation))
        # सभी शब्दों को छोटा (lowercase) करना
        image_desc = image_desc.lower()

        if image_id not in cleaned_descriptions:
            cleaned_descriptions[image_id] = []
        cleaned_descriptions[image_id].append(image_desc)

    return cleaned_descriptions

# ---- Main Program ----
filename = 'captions.txt'
# फाइल से सारा टेक्स्ट लोड करना
doc = load_doc(filename)

# कैप्शन को साफ़ करना
descriptions = clean_descriptions(doc)
print(f'\nLoaded and cleaned: {len(descriptions)} images')

# (Optional) पहले एक इमेज का साफ़ किया हुआ कैप्शन देखना
first_key = list(descriptions.keys())[0]
print("\nExample of a cleaned caption:")
print(f"Image ID: {first_key}")
print(descriptions[first_key])
