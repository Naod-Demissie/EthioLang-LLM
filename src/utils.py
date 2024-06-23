def get_unique_characters(filename: str) -> set:
    """Gets all the unique characters in a file."""
    unique_chars = set()
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            for char in line:
                unique_chars.add(char)
    return unique_chars


def get_word_counts(filename: str) -> int:
    """Counts the number of words in a file, splitting by spaces."""
    word_count = 0
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            words = line.split()
            word_count += len(words)
    return word_count


def get_word_frequencies(filename: str) -> dict:
    """Counts the frequency of each unique word in a file and returns a sorted dictionary."""
    word_frequencies = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            for word in line.split():
                if word in word_frequencies:
                    word_frequencies[word] += 1
                else:
                    word_frequencies[word] = 1
    return dict(
        sorted(word_frequencies.items(), key=lambda item: item[1], reverse=True)
    )
