# Sentiment Analysis on Product Reviews

This Python script performs sentiment analysis on product reviews stored in a CSV file. It utilizes the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool from the NLTK library.

## Prerequisites

- Python
- Pandas
- NLTK
- Matplotlib

## Installation

1. Install Python if not already installed.
2. Install required libraries using pip:

    ```bash
    pip install pandas nltk matplotlib
    ```

3. Download the NLTK Vader Lexicon:

    ```python
    import nltk
    nltk.download('vader_lexicon')
    ```

## Usage

1. Ensure your product review data is stored in a CSV file with columns for `productTitle`, `averageRating`, and `reviewDescription`.
2. Run the script:

    ```bash
    python sentiment_analysis.py
    ```

3. Follow the prompts to perform sentiment analysis on a specific product or analyze sentiment for all products.
4. The script will display sentiment analysis results including positive, negative, neutral, and compound scores for each review, as well as an overall sentiment score and recommendation based on the sentiment.
5. Additionally, a scatter plot of sentiment scores for all products will be displayed.

## File Structure

- `sentiment_analysis.py`: Python script containing sentiment analysis functionality.
- `input_data.csv`: Sample CSV file containing product review data.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
