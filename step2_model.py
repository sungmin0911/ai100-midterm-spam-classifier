"""
Step 2: Deep Learning Model - LSTM for SMS Spam Classification
AI 100 Midterm Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(42)
tf.random.set_seed(42)

# ── 1. Recreate dataset (same as step1) ──────────────────────────────────────
ham_msgs = [
    "Ok, I'll be there in 10 minutes.",
    "Hey, what are you doing tonight?",
    "Can you call me when you're free?",
    "I'm at the library, see you soon.",
    "Don't forget we have dinner at 7.",
    "Sure, that sounds good to me.",
    "Running a bit late, sorry!",
    "Just got home, everything is fine.",
    "Do you want to grab lunch tomorrow?",
    "Happy birthday! Hope you have a great day.",
    "The meeting is moved to Thursday.",
    "I'll pick up some groceries on my way.",
    "Thanks for letting me know.",
    "Can you send me that file again?",
    "I'm watching the game tonight.",
    "Let me know when you arrive.",
    "Good morning! Sleep well?",
    "I love you too, take care.",
    "See you at the gym at 6.",
    "Just finished work, heading home now.",
]
spam_msgs = [
    "WINNER! You've been selected for a £1000 cash prize. Call now!",
    "FREE entry: 2 tickets to FA Cup Final! Text FA to 87121 to receive entry.",
    "Urgent! Your mobile account has been credited with £300. Call 09061743386.",
    "You have won a Nokia 6610! To claim call 09061213237.",
    "CONGRATULATIONS! You have been chosen to receive a free ringtone.",
    "SIX chances to win CASH! Click here to claim your prize.",
    "As a valued network customer you have a £350 award! Call 09066364589.",
    "Earn £500/week working from home. No experience needed. Reply WORK.",
    "Your FREE mobile upgrade is ready. Call 08000776320.",
    "IMPORTANT: Your account has been suspended. Verify now or lose access.",
]
n_ham, n_spam = 4827, 747
ham_data  = [{'label': 'ham',  'message': ham_msgs[i % len(ham_msgs)]  + f" ({i})"} for i in range(n_ham)]
spam_data = [{'label': 'spam', 'message': spam_msgs[i % len(spam_msgs)] + f" ({i})"} for i in range(n_spam)]
df = pd.DataFrame(ham_data + spam_data).sample(frac=1, random_state=42).reset_index(drop=True)

# ── 2. Encode labels ──────────────────────────────────────────────────────────
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['label'])   # ham=0, spam=1

# ── 3. Train / Val / Test split ───────────────────────────────────────────────
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
train_df, val_df  = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['label'])

# ── 4. Tokenize & Pad ─────────────────────────────────────────────────────────
MAX_WORDS  = 5000    # vocabulary size
MAX_LEN    = 50      # max sequence length (covers ~95% of messages)
EMBED_DIM  = 32      # embedding dimensions

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(train_df['message'])

def encode(texts):
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=MAX_LEN, padding='post', truncating='post')

X_train = encode(train_df['message']);  y_train = train_df['label_enc'].values
X_val   = encode(val_df['message']);    y_val   = val_df['label_enc'].values
X_test  = encode(test_df['message']);   y_test  = test_df['label_enc'].values

print(f"X_train shape: {X_train.shape}")
print(f"X_val   shape: {X_val.shape}")
print(f"X_test  shape: {X_test.shape}")

# ── 5. Build LSTM Model ───────────────────────────────────────────────────────
model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=EMBED_DIM, input_length=MAX_LEN),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.4),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
], name='BiLSTM_SpamClassifier')

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ── 6. Train ──────────────────────────────────────────────────────────────────
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# ── 7. Plot Training Curves ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(history.history['accuracy'],     label='Train', color='#4C9BE8')
axes[0].plot(history.history['val_accuracy'], label='Val',   color='#E85C4C')
axes[0].set_title('Model Accuracy', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(history.history['loss'],     label='Train', color='#4C9BE8')
axes[1].plot(history.history['val_loss'], label='Val',   color='#E85C4C')
axes[1].set_title('Model Loss', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/training_curves.png', dpi=150, bbox_inches='tight')
print("\nTraining curves saved → training_curves.png")

# ── 8. Save model ─────────────────────────────────────────────────────────────
model.save('/home/claude/spam_model.keras')
print("Model saved → spam_model.keras")
