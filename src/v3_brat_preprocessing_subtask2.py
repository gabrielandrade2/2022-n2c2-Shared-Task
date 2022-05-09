import csv
import os
import re

import spacy

nlp = spacy.load('en_core_web_sm')

train_dir = "../trainingdata_v3/train/"
dev_dir = "../trainingdata_v3/dev/"
release_dir = "../release2/"

ann_extension = "ann"
txt_extension = "txt"

train_txt_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith("txt") and '._' not in f]
dev_txt_files = [os.path.join(dev_dir, f) for f in os.listdir(dev_dir) if f.endswith("txt") and '._' not in f]
release_txt_files = [os.path.join(release_dir, f) for f in os.listdir(release_dir) if f.endswith("txt") and '._' not in f]

train_ann_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith("ann") and '._' not in f]
dev_ann_files = [os.path.join(dev_dir, f) for f in os.listdir(dev_dir) if f.endswith("ann") and '._' not in f]
release_ann_files = [os.path.join(release_dir, f) for f in os.listdir(release_dir) if f.endswith("ann") and '._' not in f]

train_ann_files = sorted(train_ann_files)
dev_ann_files = sorted(dev_ann_files)
release_ann_files = sorted(release_ann_files)

def read_annotations(dir, csv_path):
    with open(csv_path, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        dis = 0
        count = 0
        for file_name in dir:
            # connect txt to ann
            file_txt = file_name.replace('ann', 'txt')

            # read whole with index
            with open(file_name, "r") as ann_file, open(file_txt, "r") as txt_file:
                text_old = txt_file.read()

                # get rid of "\n"
                #text = text_old.replace("\n", " ") # Here we have to add "space" or something else, otherwise this will merge some words/sentences that are split only by one (or more) \n
                text = re.sub("\n+", " ", text_old) # Gabriel: Using regex here, so we can convert multiple \n to a single "space"
                                                    # TODO: maybe we can remove "space"\n lines by using a regex such as \s?\n, to remove
                doc = nlp(text)

                line = ann_file.readline()
                while line:
                    count += 1
                    line = line.replace("\n", "")  # todo: FAITH: maybe we should replace "\n" with space. current approach results in --> avoid AnemiaCataractsODAsbestosisPeptic Ulcer DiseaseBPHNeuropathyLegs/feetChronic
                    if not line:
                        continue
                    anno = line.split("\t")
                    ann_type = anno[0]

                    if ann_type.startswith("T"):  # ex: T9	NoDisposition 3209 3219	LISINOPRIL
                        drug = anno[-1]
                        remaining = anno[1].split()
                        dis_label = remaining[0]
                        start = remaining[1]
                        end = remaining[-1]  # ex: T6\tDisposition 1293 1303;1305 1307\tnifedipine XL

                        # recalculate the span
                        #n_changeline = text_old[:int(start)].count('\n')
                        # Gabriel: The drawback of using regex before is that we need to use a slightly different one here to calculate the offset generated by the replacement of \n, which make the code slower
                        n_changeline = sum([len(match)-1 for match in re.findall("\n{2,}", text_old[:int(start)])])
                        span_start = int(start) - n_changeline
                        span_end = int(end) - n_changeline

                        # dis_label== 'Disposition', dis_label=='NoDisposition', dis_label=='Undetermined':
                        if dis_label:
                            dis += 1
                            labels = dis_label
                            sentence_spans = tuple(doc.sents)
                            word = doc.char_span(int(span_start), int(span_end), alignment_mode='expand')

                            if word != None:
                                # fetch the sentence, and before-after sentences
                                sent = word[0].sent
                                try:
                                    sent_ind = sentence_spans.index(sent)
                                    sent_ratio = sent_ind / len(sentence_spans)

                                    # decorate the metion
                                    if int(span_start) == sent.start_char:
                                        input_sent = '@' + text[int(span_start): int(span_end)] + '$' + \
                                                     text[(sent.start_char - int(span_end)): sent.end_char]

                                    elif int(span_end) == sent.end_char:
                                        input_sent = text[sent.start_char: (int(span_start))] +\
                                                     '@' + text[int(span_start): int(span_end)] + '$'
                                        # todo: it would be nicer if \n \t such in the list can be formulate in a better way
                                        # ex: 'Keep on integrilin, heparin with close PTT monitoring\n\n  \tASA\n\n  \tHolding on @BB$'

                                    else:
                                        input_sent = text[sent.start_char: (int(span_start))] + \
                                                     '@' + text[int(span_start): int(span_end)] + '$' +\
                                                     text[int(span_end): sent.end_char]

                                        # ex: 'Continue home meds (metformin + @insulin$ )'

                                    # get rid of extra spaces
                                    input_sent = input_sent.strip()

                                    # surronding sentences, special cases for first and last sentence
                                    if sent_ind == 0:
                                        input_sent = str(sentence_spans[0:3])
                                    elif sent_ind == int(len(sentence_spans) - 1):
                                        input_sent = str(sentence_spans[-3:])
                                    else:
                                        input_sent = str(sentence_spans[sent_ind - 1]) + ' ' + input_sent + ' ' + str(
                                            sentence_spans[sent_ind + 1])

                                except:
                                    import ipdb
                                    ipdb.set_trace()

                                writer.writerow(
                                    [input_sent, labels, file_name, ann_type, drug, start, end])

                            else:
                                print('word == None')
                                import ipdb
                                ipdb.set_trace()

                    line = ann_file.readline()

    print(count)
    return dis


def window_based(dir, csv_path, sorrounding_chars=100):
    with open(csv_path, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        dis = 0
        count = 0
        for file_name in dir:
            # connect txt to ann
            file_txt = file_name.replace('ann', 'txt')

            # read whole with index
            with open(file_name, "r") as ann_file, open(file_txt, "r") as txt_file:
                text_old = txt_file.read()

                # get rid of "\n"
                #text = text_old.replace("\n", " ") # Here we have to add "space" or something else, otherwise this will merge some words/sentences that are split only by one (or more) \n
                text = re.sub("\n+", " ", text_old) # Gabriel: Using regex here, so we can convert multiple \n to a single "space"
                                                    # TODO: maybe we can remove "space"\n lines by using a regex such as \s?\n, to remove
                # doc = nlp(text)

                line = ann_file.readline()
                while line:
                    count += 1
                    line = line.replace("\n", "")  # todo: FAITH: maybe we should replace "\n" with space. current approach results in --> avoid AnemiaCataractsODAsbestosisPeptic Ulcer DiseaseBPHNeuropathyLegs/feetChronic
                                                   # Gabriel: I don't think we need space in this replace here otherwise we will be adding
                    if not line:
                        continue
                    anno = line.split("\t")
                    ann_type = anno[0]

                    if ann_type.startswith("T"):  # ex: T9	NoDisposition 3209 3219	LISINOPRIL
                        drug = anno[-1]
                        remaining = anno[1].split()
                        dis_label = remaining[0]
                        start = remaining[1]
                        end = remaining[-1]  # ex: T6\tDisposition 1293 1303;1305 1307\tnifedipine XL

                        # recalculate the span
                        #n_changeline = text_old[:int(start)].count('\n')
                        # Gabriel: The drawback of using regex before is that we need to use a slightly different one here to calculate the offset generated by the replacement of \n, which make the code slower
                        n_changeline = sum([len(match)-1 for match in re.findall("\n{2,}", text_old[:int(start)])])
                        span_start = int(start) - n_changeline
                        span_end = int(end) - n_changeline

                        # dis_label== 'Disposition', dis_label=='NoDisposition', dis_label=='Undetermined':
                        if dis_label:
                            dis += 1
                            labels = dis_label

                            left_chars = sorrounding_chars
                            right_chars = sorrounding_chars

                            # find white space, so we can extract whole words
                            try:
                                while text[int(span_start) - left_chars] != " " and (int(span_start) - left_chars) > 0:
                                    left_chars += 1
                            except IndexError:
                                left_chars = 0

                            try:
                                while text[int(span_end) + sorrounding_chars] != " " and (
                                        int(span_end) + right_chars) <= len(text):
                                    right_chars += 1
                                    if text[int(span_end) + right_chars] == " ":
                                        break
                            except IndexError:
                                right_chars = len(text)-span_end #0

                            if (int(span_start)-left_chars)<0:
                                left_chars=span_start

                            # add decorator to text
                            input_sent = text[int(span_start) - left_chars:int(span_start)] + \
                                         '@' + text[int(span_start): int(span_end)] + '$' + \
                                         text[int(span_end):int(span_end) + right_chars]

                            # get rid of extra spaces
                            input_sent = input_sent.strip()

                            writer.writerow([input_sent, labels, file_name, ann_type, drug, start, end])

                    line = ann_file.readline()

    print(count)
    return dis


def main(sentence_split=True, window_split=False, sorrounding_chars=100):
    if sentence_split:
        train_path = "../sentence_split_train_p1.csv"
        dev_path = "../sentence_split_dev_p1.csv"
        dis = read_annotations(dev_ann_files, dev_path)
        dis2 = read_annotations(train_ann_files, train_path)
    if window_split:
        # train_path = "../{}-window_split_train_p1.csv".format(str(sorrounding_chars))
        # dev_path = "../{}-window_split_dev_p1.csv".format(str(sorrounding_chars))
        # dis = window_based(dev_ann_files, dev_path, sorrounding_chars)
        # dis2 = window_based(train_ann_files, train_path, sorrounding_chars)
        release_path = "../{}-window_split_200_release2.csv".format(str(sorrounding_chars))
        window_based(release_ann_files, release_path, sorrounding_chars)
if __name__ == '__main__':
    main(sentence_split=False, window_split=True,sorrounding_chars=200)