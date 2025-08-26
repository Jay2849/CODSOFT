import os
import string
import pickle
import numpy as np
from PIL import Image

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

# ======================================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
# ======================================================================================

def load_captions(filename):
    """Loads and parses the captions file."""
    with open(filename, 'r') as f:
        doc = f.read()
    mapping = {}
    for line in doc.split('\n'):
        if len(line) < 2:
            continue
        parts = line.split(',', 1)
        image_id, caption = parts[0], parts[1]
        image_id = image_id.split('.')[0]
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(caption)
    return mapping

def clean_captions(mapping):
    """Cleans captions by removing punctuation, converting to lowercase, etc."""
    table = str.maketrans('', '', string.punctuation)
    for key, descriptions in mapping.items():
        for i in range(len(descriptions)):
            desc = descriptions[i]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [w.translate(table) for w in desc]
            desc = [word for word in desc if len(word) > 1]
            desc = [word for word in desc if word.isalpha()]
            # Add start and end sequence tokens
            descriptions[i] = 'startseq ' + ' '.join(desc) + ' endseq'

def to_vocabulary(descriptions):
    """Builds a vocabulary from all unique words in the captions."""
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc

def save_descriptions(descriptions, filename):
    """Saves the cleaned descriptions to a file."""
    lines = []
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    with open(filename, 'w') as f:
        f.write(data)

# ======================================================================================
# SECTION 2: IMAGE FEATURE EXTRACTION (using VGG16)
# ======================================================================================

def extract_features(directory):
    """Extracts features from each image in the directory using VGG16."""
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    print("VGG16 Model Summary:")
    print(model.summary())
    
    features = dict()
    for name in os.listdir(directory):
        filename = os.path.join(directory, name)
        try:
            image = load_img(filename, target_size=(224, 224))
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)
            feature = model.predict(image, verbose=0)
            image_id = name.split('.')[0]
            features[image_id] = feature
        except Exception as e:
            print(f"Error processing {name}: {e}")
    return features

# ======================================================================================
# SECTION 3: DATA PREPARATION FOR TRAINING
# ======================================================================================

def load_clean_descriptions(filename, dataset_keys):
    """Loads cleaned descriptions for a specific dataset (e.g., train/test)."""
    with open(filename, 'r') as f:
        doc = f.read()
    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        if image_id in dataset_keys:
            if image_id not in descriptions:
                descriptions[image_id] = []
            desc = ' '.join(image_desc)
            descriptions[image_id].append(desc)
    return descriptions

def create_sequences(tokenizer, max_length, desc_list, photo, vocab_size):
    """Creates input-output sequence pairs for training."""
    X1, X2, y = [], [], []
    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

def max_length(descriptions):
    """Calculates the max length of any caption."""
    lines = [d for key in descriptions for d in descriptions[key]]
    return max(len(d.split()) for d in lines)

def data_generator(descriptions, photos, tokenizer, max_length, vocab_size):
    """A generator function to yield batches of data, saving memory."""
    while 1:
        for key, desc_list in descriptions.items():
            if key in photos:
                photo = photos[key][0]
                in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo, vocab_size)
                yield [in_img, in_seq], out_word

# ======================================================================================
# SECTION 4: MODEL DEFINITION
# ======================================================================================

def define_model(vocab_size, max_length):
    """Defines the image captioning model."""
    # Feature extractor model
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # Sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    print("\nFinal Model Summary:")
    print(model.summary())
    # plot_model(model, to_file='model.png', show_shapes=True)
    return model

# ======================================================================================
# SECTION 5: INFERENCE (CAPTION GENERATION)
# ======================================================================================

def word_for_id(integer, tokenizer):
    """Maps an integer to a word."""
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_caption(model, tokenizer, photo, max_length):
    """Generates a caption for an image."""
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    
    # Clean up the final caption
    final_caption = in_text.split()
    final_caption = final_caption[1:-1] # Remove startseq and endseq
    final_caption = ' '.join(final_caption)
    return final_caption


# ======================================================================================
# MAIN EXECUTION SCRIPT
# ======================================================================================

if __name__ == '__main__':
    # --- 1. PREPARE TEXT DATA ---
    captions_file = 'captions.txt'
    descriptions = load_captions(captions_file)
    print(f'Loaded {len(descriptions)} images with captions.')
    
    clean_captions(descriptions)
    
    # Summarize vocabulary
    vocabulary = to_vocabulary(descriptions)
    print(f'Original Vocabulary Size: {len(vocabulary)}')

    # For simplicity, we use all images for training.
    # In a real project, you should create a train/test split.
    train_keys = list(descriptions.keys())
    
    # Save cleaned captions to a file
    save_descriptions(descriptions, 'descriptions.txt')

    # --- 2. EXTRACT IMAGE FEATURES ---
    # !! IMPORTANT: Change this to your actual images folder path
    images_directory = 'images' 
    features_file = 'features.pkl'

    if not os.path.exists(features_file):
        print("Image features not found, starting extraction...")
        features = extract_features(images_directory)
        print(f'Extracted Features for {len(features)} images.')
        # Save features
        with open(features_file, 'wb') as f:
            pickle.dump(features, f)
        print(f"Features saved to {features_file}")
    else:
        print(f"Loading image features from {features_file}...")
        with open(features_file, 'rb') as f:
            features = pickle.load(f)
        print(f"Loaded features for {len(features)} images.")

    # --- 3. PREPARE DATA FOR MODEL ---
    # Load descriptions for training
    train_descriptions = {k: descriptions[k] for k in train_keys if k in features}
    
    # Prepare tokenizer
    all_desc_list = [d for key in train_descriptions for d in train_descriptions[key]]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_desc_list)
    vocab_size = len(tokenizer.word_index) + 1
    print(f'Final Vocabulary Size: {vocab_size}')

    # Determine the maximum sequence length
    max_len = max_length(train_descriptions)
    print(f'Max Caption Length: {max_len}')
    
    # --- 4. DEFINE AND TRAIN MODEL ---
    model = define_model(vocab_size, max_len)
    
    epochs = 20 # Number of training rounds
    steps_per_epoch = len(train_descriptions)
    
    # Create the data generator
    generator = data_generator(train_descriptions, features, tokenizer, max_len, vocab_size)
    
    print("\n--- Starting Model Training ---")
    print(f"Epochs: {epochs}")
    print(f"Steps per Epoch: {steps_per_epoch}")
    
    # Fit model
    model.fit(generator, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=1)
    
    # Save the trained model
    model.save('image_captioning_model.h5')
    # Save the tokenizer
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    print("\n--- Training Complete ---")
    print("Model saved to 'image_captioning_model.h5'")
    print("Tokenizer saved to 'tokenizer.pkl'")

    # --- 5. EXAMPLE OF INFERENCE (OPTIONAL) ---
    # You can run this part separately after training is done.
    # from tensorflow.keras.models import load_model
    # model = load_model('image_captioning_model.h5')
    # with open('tokenizer.pkl', 'rb') as f:
    #     tokenizer = pickle.load(f)
    
    # # Provide a sample image feature to test
    # sample_image_key = list(features.keys())[0] # Taking the first image as an example
    # photo_feature = features[sample_image_key]
    
    # generated = generate_caption(model, tokenizer, photo_feature, max_len)
    # print("\n--- Example Prediction ---")
    # print(f"Image: {sample_image_key}.jpg")
    # print(f"Generated Caption: {generated}")