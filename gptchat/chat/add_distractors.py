import numpy
import sys


def main(num_distractors):
    if num_distractors <= 0:
        raise Exception("num_distractors should be larger than 0")

    texts = []
    replies = []
    for line in sys.stdin:
        text, reply = line.strip("\n").split("\t")
        texts.append(text)
        replies.append(reply)

    random_vals = zip(
        *[numpy.random.permutation(len(texts))
          for _ in range(num_distractors)]
    )
    for i, rand in enumerate(random_vals):
        items = [texts[i], replies[i]] + [replies[j] for j in rand]
        print("\t".join(items))


if __name__ == "__main__":
    import fire

    fire.Fire(main)
