import matplotlib.pyplot as plt
from wordcloud import WordCloud


def clean_text_file(input_path: str, output_path: str):
    """Cleans a text file by removing unwanted characters and saving the cleaned text."""
    with open(input_path, "r", encoding="utf-8") as infile, open(
        output_path, "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            # Example cleaning: remove extra spaces and convert to lowercase
            cleaned_line = " ".join(line.strip().split()).lower()
            outfile.write(cleaned_line + "\n")


def split_dataset(
    input_path: str, train_path: str, test_path: str, train_ratio: float = 0.8
):
    """Splits a text dataset into training and testing sets."""
    with open(input_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    split_index = int(len(lines) * train_ratio)
    train_lines = lines[:split_index]
    test_lines = lines[split_index:]

    with open(train_path, "w", encoding="utf-8") as train_file:
        train_file.writelines(train_lines)

    with open(test_path, "w", encoding="utf-8") as test_file:
        test_file.writelines(test_lines)


def generate_wordcloud(word_frequencies: dict, font_path: str, output_path: str = None):
    """Generates and displays a word cloud from word frequencies."""
    wordcloud = WordCloud(
        font_path=font_path,
        width=1200,
        height=600,
        max_words=len(word_frequencies),
        background_color="white",
    ).generate_from_frequencies(word_frequencies)

    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    if output_path:
        plt.savefig(output_path)
    plt.show()


def get_training_corpus(filename: str, batch_size: int = 1000):
    """Loads a text file in batches and yields lists of text."""
    with open(filename, "r", encoding="utf-8") as f:
        while True:
            lines = [next(f) for _ in range(batch_size)]
            if not lines:
                break
            yield lines
