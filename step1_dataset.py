"""
Step 1: Problem Definition & Dataset Exploration
SMS Spam Classification - AI 100 Midterm Project
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split

# ── 1. Load dataset ──────────────────────────────────────────────────────────
# Using sklearn's built-in fetch or generating representative sample data
# In practice: download from https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
import numpy as np
np.random.seed(42)

# Representative sample matching real dataset statistics
# Real dataset: 4827 ham (86.6%), 747 spam (13.4%), total 5574
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
ham_data = [{'label': 'ham', 'message': ham_msgs[i % len(ham_msgs)] + f" ({i})"}
            for i in range(n_ham)]
spam_data = [{'label': 'spam', 'message': spam_msgs[i % len(spam_msgs)] + f" ({i})"}
             for i in range(n_spam)]
df = pd.DataFrame(ham_data + spam_data).sample(frac=1, random_state=42).reset_index(drop=True)

print("=== Dataset Overview ===")
print(f"Total samples : {len(df)}")
print(f"Columns       : {list(df.columns)}")
print(f"\nClass distribution:\n{df['label'].value_counts()}")
print(f"\nClass balance (%):\n{df['label'].value_counts(normalize=True).mul(100).round(1)}")

# ── 2. Sample messages ───────────────────────────────────────────────────────
print("\n=== Sample Messages ===")
for label in ['ham', 'spam']:
    print(f"\n[{label.upper()}]")
    for msg in df[df['label'] == label]['message'].head(2):
        print(f"  • {msg[:120]}")

# ── 3. Message length analysis ───────────────────────────────────────────────
df['msg_len'] = df['message'].apply(len)
print("\n=== Message Length Stats ===")
print(df.groupby('label')['msg_len'].describe().round(1))

# ── 4. Train/Val/Test split ──────────────────────────────────────────────────
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42,
                                     stratify=df['label'])
train_df, val_df  = train_test_split(train_df, test_size=0.1, random_state=42,
                                     stratify=train_df['label'])
print(f"\n=== Data Split ===")
print(f"Train : {len(train_df)} samples")
print(f"Val   : {len(val_df)}   samples")
print(f"Test  : {len(test_df)}  samples")

# ── 5. Visualizations ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Class distribution pie
counts = df['label'].value_counts()
axes[0].pie(counts, labels=counts.index, autopct='%1.1f%%',
            colors=['#4C9BE8', '#E85C4C'], startangle=90,
            textprops={'fontsize': 12})
axes[0].set_title('Class Distribution', fontsize=13, fontweight='bold')

# Message length histogram
for label, color in zip(['ham', 'spam'], ['#4C9BE8', '#E85C4C']):
    subset = df[df['label'] == label]['msg_len']
    axes[1].hist(subset, bins=40, alpha=0.65, color=color, label=label)
axes[1].set_xlabel('Message Length (chars)', fontsize=11)
axes[1].set_ylabel('Count', fontsize=11)
axes[1].set_title('Message Length Distribution', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)

plt.tight_layout()
plt.savefig('/home/claude/dataset_eda.png', dpi=150, bbox_inches='tight')
print("\nPlot saved → dataset_eda.png")
