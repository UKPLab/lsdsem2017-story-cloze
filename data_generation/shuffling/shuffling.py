import csv
import sys
from random import shuffle, randrange

LABEL_POSITIVE = 1
LABEL_NEGATIVE = 0

SHUFFLED_CSV_HEADER = ["storyid", "sentence1", "sentence2", "sentence3", "sentence4", "ending", "label"]
ROCSTORIES_CSV_HEADER = ["storyid", "storytitle", "sentence1", "sentence2", "sentence3", "sentence4", "sentence5"]


def create_shuffled_rocstories(src_path, dest_path):
    """
    Shuffles the story context, inserts the ending randomly between the first and fourth sentence position.
    :param src_path: the path to the original ROCStories CSV file
    :param dest_path: the destination for the processed ROCStories CSV file
    :return:
    """
    with open(src_path, 'r') as src, open(dest_path, 'w') as dest:
        reader = csv.reader(src, delimiter=',', quotechar='"')
        writer = csv.writer(dest, delimiter=',', quotechar='"')

        writer.writerow(SHUFFLED_CSV_HEADER)

        reader = iter(reader)
        next(reader)  # skip CSV header
        for row in reader:
            story_id = row[0]
            story_context = row[2:6]
            story_ending = row[6]

            # write the "correct" version of the story
            writer.writerow([story_id] + story_context + [story_ending] + [LABEL_POSITIVE])

            # shuffle the story context, insert the ending randomly between the first and fourth sentence position
            shuffle(story_context)
            ending_insert_index = randrange(len(story_context))
            story_context.insert(ending_insert_index, story_ending)
            shuffled_story = story_context

            # write the artificial negative training example version of the story
            writer.writerow([story_id] + shuffled_story + [LABEL_NEGATIVE])

if __name__ == '__main__':
    # source, destination
    create_shuffled_rocstories(*sys.argv[1:])