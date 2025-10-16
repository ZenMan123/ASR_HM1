from collections import defaultdict

import editdistance


def calc_wer(target_text, predicted_text) -> float:
    if not target_text:
        return 1 if predicted_text else 0
    return editdistance.eval(target_text.split(), predicted_text.split()) / len(
        target_text.split()
    )


def calc_cer(target_text, predicted_text) -> float:
    if not target_text:
        return 1 if predicted_text else 0
    return editdistance.eval(target_text, predicted_text) / len(target_text)


def expand_and_merge_beams(dp, cur_step_prob, vocab, empty_tok):
    new_dp = defaultdict(float)

    for (pref, prev_char), pref_proba in dp.items():
        for idx, char in enumerate(vocab):
            cur_proba = pref_proba * cur_step_prob[idx]
            cur_char = char

            if char == empty_tok:
                cur_pref = pref
            else:
                if prev_char != char:
                    cur_pref = pref + char
                else:
                    cur_pref = pref

            new_dp[(cur_pref, cur_char)] += cur_proba

    return new_dp


def truncate_beams(dp, beam_size):
    items = sorted(list(dp.items()), key=lambda x: -x[1])[:beam_size]
    return dict(items)
