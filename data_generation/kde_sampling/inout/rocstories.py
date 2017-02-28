import csv
import random

STORY_CLOZE_CSV_HEADER = ["InputStoryid", "InputSentence1", "InputSentence2", "InputSentence3", "InputSentence4",
                          "RandomFifthSentenceQuiz1", "RandomFifthSentenceQuiz2", "AnswerRightEnding"]


def write_rocstories_with_negative_endings(dest_path, rocst_path, wrong_ending_for_story_id):
    with open(dest_path, 'w') as dest, open(rocst_path, 'r') as rocstories:
        reader = csv.reader(rocstories, delimiter=',', quotechar='"')
        writer = csv.writer(dest, delimiter=',', quotechar='"')

        writer.writerow(STORY_CLOZE_CSV_HEADER)
        reader = iter(reader)
        next(reader)  # skip the ROCStories CSV header

        for row in reader:
            story_id = row[0]
            story_context = row[2:6]
            correct_end = row[6]
            wrong_end = wrong_ending_for_story_id[story_id]

            # randomly shuffle the two endings (you never know...)
            endings = [correct_end, wrong_end, 1] if random.choice([True, False]) else [wrong_end, correct_end, 2]
            writer.writerow([story_id] + story_context + endings)