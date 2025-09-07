import tensorflow as tf
from datasets import load_dataset
from transformers import DistilBertTokenizer, TFDistilBertModel
from pathlib import Path


# ---------------------------- Data Preparation ---------------------------- #
def load_and_prepare_dataset():
    """Load the SNIPS intent dataset and map intent labels to integers."""
    ds = load_dataset("bkonkle/snips-joint-intent")
    train, test = ds['train'], ds['test']
    train_pd = train.to_pandas()

    # Map intents to indices
    labels_map = {intent: idx for idx, intent in enumerate(train_pd['intent'].unique())}
    train_pd['intent'] = train_pd['intent'].map(lambda x: labels_map[x])

    return train_pd, labels_map


def split_dataset(df, train_split=10_000, valid_split=12_000):
    """Split dataset into train, validation, and test sets."""
    features, labels = df['input'], df['intent']
    return (
        features[:train_split], labels[:train_split],
        features[train_split:valid_split], labels[train_split:valid_split],
        features[valid_split:], labels[valid_split:]
    )


# ---------------------------- Model Definition ---------------------------- #
class DistilBERTClassifier(tf.keras.Model):
    def __init__(self, transformer, num_labels):
        super().__init__()
        self.transformer = transformer
        self.classifier = tf.keras.layers.Dense(num_labels, activation='softmax')

    def call(self, inputs):
        output = self.transformer(inputs)[0]  # last_hidden_state
        cls_token = output[:, 0, :]  # [CLS] token
        return self.classifier(cls_token)


# ---------------------------- Training & Tokenization ---------------------------- #
def tokenize_texts(tokenizer, texts):
    """Tokenize a list of texts for DistilBERT."""
    return tokenizer(
        texts.to_list(),
        truncation=True,
        padding=True,
        return_tensors='tf'
    )


def train_model(model, tokenizer, X_train, y_train, X_valid, y_valid, epochs=1, batch_size=64):
    """Train the DistilBERT classifier."""
    train_inputs = tokenize_texts(tokenizer, X_train)
    valid_inputs = tokenize_texts(tokenizer, X_valid)

    y_train_tensor = tf.convert_to_tensor(y_train.to_list())
    y_valid_tensor = tf.convert_to_tensor(y_valid.to_list())

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    history = model.fit(
        x={'input_ids': train_inputs['input_ids'], 'attention_mask': train_inputs['attention_mask']},
        y=y_train_tensor,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(
            {'input_ids': valid_inputs['input_ids'], 'attention_mask': valid_inputs['attention_mask']},
            y_valid_tensor
        )
    )
    return history


# ---------------------------- Model Wrapper ---------------------------- #
class FillerSTransformer:
    def __init__(self, transformer, num_labels, model_path="filler_s_transformer.h5"):
        self.transformer = transformer
        self.num_labels = num_labels
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        if Path(self.model_path).is_file():
            temp_model = DistilBERTClassifier(self.transformer, self.num_labels)
            # Build weights with dummy input
            dummy_input_ids = tf.ones((1, 10), dtype=tf.int32)
            dummy_attention_mask = tf.ones((1, 10), dtype=tf.int32)
            temp_model({'input_ids': dummy_input_ids, 'attention_mask': dummy_attention_mask})
            temp_model.load_weights(self.model_path)
            return temp_model
        else:
            return None

    def predict(self, tokenizer, text):
        inputs = tokenizer(text, truncation=True, padding=True, return_tensors='tf')
        return self.model.predict({'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']})


# ---------------------------- Main Execution ---------------------------- #
def main():
    train_df, labels_map = load_and_prepare_dataset()
    X_train, y_train, X_valid, y_valid, X_test, y_test = split_dataset(train_df)

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    transformer = TFDistilBertModel.from_pretrained("distilbert-base-uncased")

    model = DistilBERTClassifier(transformer, num_labels=len(labels_map))
    train_model(model, tokenizer, X_train, y_train, X_valid, y_valid, epochs=1)

    # Save model weights
    model.save_weights("filler_s_transformer.h5")

    # Reload and test
    fillers_model = FillerSTransformer(transformer, len(labels_map))
    print(fillers_model.predict(tokenizer, ["what you say?"]))


if __name__ == "__main__":
    main()
