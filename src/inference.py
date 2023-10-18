import pandas as pd
import torch
from tqdm import trange


def prepare_sample(nlp, text):
    doc = nlp(text)
    sample_df = {"idx_sent_in_doc": [],
                 "complex_sent": [],
                 "is_simple_detected": [],
                 "simplified_sent": []}

    for idx_s, sent in enumerate(doc.sentences):
        simplified_part, is_already_simple = "N/A", False
        sample_df["idx_sent_in_doc"].append(idx_s)
        sample_df["complex_sent"].append(sent.text)
        sample_df["is_simple_detected"].append(is_already_simple)
        sample_df["simplified_sent"].append(simplified_part)

    sample_df = pd.DataFrame(sample_df)

    return sample_df


def cls(cls_tokenizer, cls_model, cls_device, sample_df):
    encoded_complex = cls_tokenizer(sample_df["complex_sent"].tolist(),
                                    max_length=65,
                                    truncation=True,
                                    padding="max_length",
                                    return_tensors="pt")
    cls_batch_size = 1024
    num_batches = (sample_df.shape[0] + cls_batch_size - 1) // cls_batch_size

    are_sents_simple = []
    with torch.inference_mode():
        for idx_batch in trange(num_batches):
            s_b, e_b = idx_batch * cls_batch_size, (idx_batch + 1) * cls_batch_size

            input_data = {_k: _v[s_b: e_b].to(cls_device) for _k, _v in encoded_complex.items()}
            logits = cls_model(**input_data).logits
            probas = torch.softmax(logits, dim=-1).cpu()
            preds = torch.argmax(probas, dim=-1)

            are_sents_simple.extend(preds.bool().tolist())

    sample_df["is_simple_detected"] = are_sents_simple

    return sample_df


def gen(gen_tokenizer, gen_model, sample_df):
    prefix = "simplify: "

    def handle_sentence(input_sentence, is_simple_detected=False):
        simplified = str(input_sentence)

        if not is_simple_detected:
            input_ids = gen_tokenizer(f"{prefix} {input_sentence}", return_tensors="pt", max_length=128,
                                      truncation=True).input_ids.to(gen_model.device)
            outputs = gen_model.generate(input_ids,
                                         max_length=128,
                                         num_beams=5,
                                         do_sample=True,
                                         top_k=5,
                                         temperature=0.7
                                         )

            simplified = gen_tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)

        return simplified

    simplified_sents = []
    with torch.inference_mode():
        for idx_ex in trange(sample_df.shape[0]):
            curr_input_str = sample_df.iloc[idx_ex]["complex_sent"]
            mask = sample_df.iloc[idx_ex]["is_simple_detected"]

            simplified_sents.append(handle_sentence(curr_input_str, is_simple_detected=mask))

    sample_df = pd.DataFrame(sample_df)
    sample_df["simplified_sent"] = simplified_sents

    return sample_df


def add_stats(nlp, text, textbook, general):
    doc = nlp(text)
    num_of_words = 0
    num_of_sentences = 0
    num_of_characters = len(text)
    words = []
    words_on_textbook = 0
    words_on_general = 0
    general_words = {}
    textbook_words = {}
    for sent in doc.sentences:
        num_of_sentences += 1
        for word in sent.words:
            if word.pos != 'PUNCT':
                num_of_words += 1  # count words
                words_on_textbook += 1 if word.lemma in textbook else 0
                words_on_general += 1 if word.lemma in general else 0
                words.append(word.text)
                textbook_words[word.text] = True if word.lemma in textbook else False
                general_words[word.text] = True if word.lemma in general else False

    percent_not_on_general = round((1 - (words_on_general / num_of_words)) * 100, 2)
    percent_not_on_textbook = round((1 - (words_on_textbook / num_of_words)) * 100, 2)

    return (num_of_words,
            num_of_sentences,
            num_of_characters,
            percent_not_on_general,
            percent_not_on_textbook,
            general_words,
            textbook_words)


def df2json(nlp, sample_df, textbook, general):
    complex_text = ' '.join(sample_df['complex_sent'].tolist())
    (complex_num_of_words,
     complex_num_of_sentences,
     complex_num_of_characters,
     complex_percent_not_on_general,
     complex_percent_not_on_textbook,
     complex_general_words,
     complex_textbook_words) = add_stats(nlp, complex_text, textbook, general)

    simplified_text = ' '.join(sample_df['simplified_sent'].tolist())
    (simplified_num_of_words,
     simplified_num_of_sentences,
     simplified_num_of_characters,
     simplified_percent_not_on_general,
     simplified_percent_not_on_textbook,
     simplified_general_words,
     simplified_textbook_words) = add_stats(nlp, simplified_text, textbook, general)

    return {
        'complex_text': complex_text,
        'complex_sentences': sample_df['complex_sent'].tolist(),
        'complex_num_of_words': complex_num_of_words,
        'complex_num_of_sentences': complex_num_of_sentences,
        'complex_num_of_characters': complex_num_of_characters,
        'complex_percent_not_on_general': complex_percent_not_on_general,
        'complex_percent_not_on_textbook': complex_percent_not_on_textbook,
        'complex_general_words': complex_general_words,
        'complex_textbook_words': complex_textbook_words,

        'simplified_text': simplified_text,
        'simplified_sentences': sample_df['simplified_sent'].tolist(),
        'simplified_num_of_words': simplified_num_of_words,
        'simplified_num_of_sentences': simplified_num_of_sentences,
        'simplified_num_of_characters': simplified_num_of_characters,
        'simplified_percent_not_on_general': simplified_percent_not_on_general,
        'simplified_percent_not_on_textbook': simplified_percent_not_on_textbook,
        'simplified_general_words': simplified_general_words,
        'simplified_textbook_words': simplified_textbook_words
    }


def refresh_stats(nlp, text, textbook, general):
    (num_of_words,
     num_of_sentences,
     num_of_characters,
     percent_not_on_general,
     percent_not_on_textbook,
     general_words,
     textbook_words) = add_stats(nlp, text, textbook, general)

    return {
        'num_of_words': num_of_words,
        'num_of_sentences': num_of_sentences,
        'num_of_characters': num_of_characters,
        'percent_not_on_general': percent_not_on_general,
        'percent_not_on_textbook': percent_not_on_textbook,
        'general_words': general_words,
        'textbook_words': textbook_words,
    }


def get_long_words(nlp, text, length):
    doc = nlp(text)
    long_words = []
    for sent in doc.sentences:
        for word in sent.words:
            if len(word.text) > length:
                long_words.append(word.text)
    return long_words
