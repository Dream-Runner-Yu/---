class WordsFrequency:

    def __init__(self, book: List[str]):
        # self.data = Counter(book)
        self.data = {}
        for i in book:
            if i not in self.data:
                self.data[i] = 1
            else:
                self.data[i] += 1


    def get(self, word: str) -> int:
        return self.data[word] if word in self.data else 0



# Your WordsFrequency object will be instantiated and called as such:
# obj = WordsFrequency(book)
# param_1 = obj.get(word)