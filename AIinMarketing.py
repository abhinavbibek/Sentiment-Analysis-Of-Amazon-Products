import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

nltk.download('vader_lexicon')

file = 'input_data.csv'
dfs = pd.read_csv(file)

product_sentiments = {}

# Plot count of reviews by stars
ax = dfs['averageRating'].value_counts().sort_index().plot(kind='bar',
                                                          title='Count of Reviews by Stars',
                                                          figsize=(10, 5))
ax.set_xlabel('Review Stars')
ax.set_ylabel('Number of Products Sold')  # Add ylabel for count of products

# Annotate each bar with the count of products
for i, count in enumerate(dfs['averageRating'].value_counts().sort_index()):
    ax.text(i, count, str(count), ha='center', va='bottom')
plt.show()


def analyze_product_sentiment(product_name):
    matching_products = dfs[dfs['productTitle'].str.lower().str.contains(product_name.lower())]['productTitle'].unique()

    if len(matching_products) == 0:
        print(f"No products found matching '{product_name}'.")
    elif len(matching_products) == 1:
        product_reviews = dfs[(dfs['productTitle'].str.lower() == matching_products[0].lower()) & (dfs['reviewDescription'].notnull())]['reviewDescription'].tolist()
        calculate_sentiment(matching_products[0], product_reviews)
    else:
        print("Multiple products found matching your input:")
        for idx, product in enumerate(matching_products, start=1):
            print(f"{idx}. {product}")
        choice = input("Enter the number of the specific product you want to analyze: ")
        if choice.isdigit() and int(choice) in range(1, len(matching_products) + 1):
            chosen_product = matching_products[int(choice) - 1]
            product_reviews = dfs[(dfs['productTitle'].str.lower() == chosen_product.lower()) & (dfs['reviewDescription'].notnull())]['reviewDescription'].tolist()
            calculate_sentiment(chosen_product, product_reviews)
        else:
            print("Invalid choice.")

def display_reviews(reviews):
    print("Here are some of the reviews of the product you searched for:\n")
    for review in reviews:
        print("+" + "-"*78 + "+")  # Border
        print("| {:^76} |".format(review))  # Review content
        print("+" + "-"*78 + "+")  # Border
    print("\n")

def calculate_sentiment(product_name, product_reviews):
    sid = SentimentIntensityAnalyzer()
    product_scores = []

    print(f"\nSentiment analysis for product '{product_name}':\n")
    
    for review in product_reviews:
        ss = sid.polarity_scores(review)
        print(f"Review: {review}")
        print(f"Negative Score: {ss['neg']:.4f}")
        print(f"Neutral Score: {ss['neu']:.4f}")
        print(f"Positive Score: {ss['pos']:.4f}")
        print(f"Compound Score: {ss['compound']:.4f}")
        print('-' * 111)

        product_scores.append(ss['compound'])

    avg_score = sum(product_scores) / len(product_scores)
    product_sentiments[product_name] = avg_score

    print(f"Average Sentiment Score: {avg_score:.4f}")

    if avg_score >= 0.05:
        comment = "It's recommended to buy this product."
    elif avg_score <= -0.05:
        comment = "It's not recommended to buy this product."
    else:
        comment = "You may consider buying this product based on other factors."

    print(f"Conclusion of Product: {comment}\n")



def analyze_all_products_sentiment():
    sid = SentimentIntensityAnalyzer()
    
    for product_name in dfs['productTitle'].unique():
        product_reviews = dfs[(dfs['productTitle'] == product_name) & (dfs['reviewDescription'].notnull())]['reviewDescription'].tolist()
        
        if product_reviews:
            product_scores = []
            
            for review in product_reviews:
                ss = sid.polarity_scores(review)
                product_scores.append(ss['compound'])
            
            avg_score = sum(product_scores) / len(product_scores)
            product_sentiments[product_name] = avg_score

def plot_all_products_sentiments():
    total_products = len(product_sentiments)
    print(f"Total number of Unique products: {total_products}")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(product_sentiments)), list(product_sentiments.values()), c=list(product_sentiments.values()), cmap='coolwarm')
    plt.xlabel('')  # Empty x-label
    plt.ylabel('Sentiment Score')
    plt.title('Sentiment Analysis of Products')
    plt.colorbar(label='Sentiment Score')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



product_name = input("Enter the product name for which you want reviews: ")

analyze_product_sentiment(product_name)

analyze_all_products_sentiment()
plot_all_products_sentiments()
