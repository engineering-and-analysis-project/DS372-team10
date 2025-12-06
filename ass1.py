import pandas as pd
import matplotlib.pyplot as plt

# =======================
# 1. Load the datasets
# =======================
payments = pd.read_csv('olist_order_payments_dataset.csv')
reviews = pd.read_csv('olist_order_reviews_dataset.csv')

# =======================
# 2. Convert date columns
# =======================
for col in ['review_creation_date', 'review_answer_timestamp']:
    reviews[col] = pd.to_datetime(reviews[col], errors='coerce')

# Compute response time
reviews['review_response_time_hours'] = (
    (reviews['review_answer_timestamp'] - reviews['review_creation_date'])
    .dt.total_seconds() / 3600
)

# Flag missing comment
reviews['is_comment_missing'] = reviews['review_comment_message'].isna()

# =======================
# 3. Aggregate payments
# =======================
payments_agg = payments.groupby('order_id').agg(
    total_payment=('payment_value', 'sum'),
    num_payments=('payment_value', 'count'),
    avg_payment=('payment_value', 'mean'),
    max_payment=('payment_value', 'max'),
).reset_index()

# =======================
# 4. Merge with reviews
# =======================
reviews_first = reviews.sort_values('review_creation_date') \
                       .drop_duplicates('order_id', keep='first')

merged = payments_agg.merge(
    reviews_first[['order_id', 'review_score', 'review_comment_message',
                   'review_response_time_hours']],
    on='order_id', how='left'
)

# Fill missing review scores (if any)
merged['review_score'] = merged['review_score'].fillna(0)

# =======================
# 5. PLOTS
# =======================

# -------- Plot 1: Review Score Histogram -------- #
plt.figure()
merged['review_score'].plot(kind='hist')
plt.title('Review Score Distribution')
plt.xlabel('Review Score')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# -------- Plot 2: Total Payment Histogram -------- #
plt.figure()
merged['total_payment'].plot(kind='hist')
plt.title('Total Payment Distribution')
plt.xlabel('Total Payment')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# -------- Plot 3: Avg Payment vs Review Score Scatter -------- #
plt.figure()
plt.scatter(merged['avg_payment'], merged['review_score'])
plt.title('Avg Payment vs Review Score')
plt.xlabel('Average Payment')
plt.ylabel('Review Score')
plt.tight_layout()
plt.show()

# -------- Plot 4: Number of Payments Histogram -------- #
plt.figure()
merged['num_payments'].plot(kind='hist')
plt.title('Number of Payments Distribution')
plt.xlabel('Number of Payments')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

print("All charts generated successfully!")
