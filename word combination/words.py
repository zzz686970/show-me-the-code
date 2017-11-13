from collections import defaultdict

class Words:
    def __init__(self, words_file = '/usr/local/words'):
        with open(words_file) as f:
            word_length = defaultdict(list)
            for word in f:
                word = word.rstrip('\n')
                if not word.isalpha() or not word.islower():
                    continue
                word_length[len(word)].append(word)


        word_length[1] = list('IOa')
        self.word_length = word_length

    def possibilities(self, num_letters, num_words):
        word_length = self.word_length
        if num_words == 1:
            for word1 in word_length[num_letters]:
                yield (word1,)
            return

        for len in range(1, num_letters-num_words+2):
            for word in word_length[len]:
                for rest in self.possibilities(num_letters-len, num_words-1):
                    yield (word,) + rest


def main():
    words = Words()
    for sentence in words.possibilities(8,3):
        print(sentence)

if __name__ == '__main__':
    main()
