import os
import sys
import time
import codecs
import requests
import numpy as np
import xmlrpc.client as xc
from flask import Flask, render_template, request
from flask_pymongo import PyMongo

app = Flask(__name__)

MAX_LENGTH=50   # the maximum sentence length of word alignment visualization
MAX_TERMS=15   # the maximum number of dictionary terms
punctuations = ['!', '"', '#', '$', '%', '&', "\\", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>',
                '?', '@', '[', '\\\\', ']', '^', '_', '`', '{', '|', '}', '~', "'", "``", "''"]
punctuations = set(punctuations)
stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
             "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
             "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
             "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
             "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as",
             "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through",
             "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
             "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how",
             "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
             "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
stopwords = set(stopwords)
OLD_ENGLISH = {"thy": "your", "thou": "you", "Thy": "Your", "Thou": "You"}

# moses tokenizer
from sacremoses import MosesTruecaser, MosesTokenizer, MosesDetokenizer, MosesDetruecaser
mtok = MosesTokenizer(lang='en')
mtr = MosesTruecaser("vocab/truecase-model.en")
md = MosesDetokenizer(lang="en")
mdtr = MosesDetruecaser()

# bpe tokenizer
from subword_nmt.apply_bpe import BPE, read_vocabulary
vocabulary = read_vocabulary(codecs.open("vocab/vocab.bpe35000.chr", encoding='utf-8'), 10)
bpe = BPE(codes=codecs.open("vocab/codes_file_chr_35000", encoding='utf-8'), merges=35000, vocab=vocabulary)

# load nmt models
import onmt.opts
from translator_for_demo import build_translator
from onmt.utils.parse import ArgumentParser


def _parse_opt(opt):
    prec_argv = sys.argv
    sys.argv = sys.argv[:1]
    parser = ArgumentParser()
    onmt.opts.translate_opts(parser)

    opt['src'] = "dummy_src"
    opt['replace_unk'] = True

    for (k, v) in opt.items():
        if k == 'models':
            sys.argv += ['-model']
            sys.argv += [str(model) for model in v]
        elif type(v) == bool:
            sys.argv += ['-%s' % k]
        else:
            sys.argv += ['-%s' % k, str(v)]

    opt = parser.parse_args()
    ArgumentParser.validate_translate_opts(opt)
    opt.cuda = opt.gpu > -1

    sys.argv = prec_argv
    return opt


enchr_opt = {'models': ["models/data_demo_feedback_round2_ced_enchr_min0_brnn_adam_0.0005_1000_2_0.5_1024_LSTM_0.2_seed7/model_step_0_release.pt",
                        "models/data_demo_feedback_round2_ced_enchr_min0_brnn_adam_0.0005_1000_2_0.5_1024_LSTM_0.2_seed77/model_step_0_release.pt",
                        "models/data_demo_feedback_round2_ced_enchr_min0_brnn_adam_0.0005_1000_2_0.5_1024_LSTM_0.2_seed777/model_step_0_release.pt"]}
enchr_opt = _parse_opt(enchr_opt)
enchr_translator = build_translator(enchr_opt, report_score=False)

chren_opt = {'models': ["models/data_demo_feedback_round2_ced_chren_bpe35000-35000_min10_brnn_adam_0.0005_1000_2_0.3_1024_LSTM_0.2_7/model_step_0_release.pt",
                        "models/data_demo_feedback_round2_ced_chren_bpe35000-35000_min10_brnn_adam_0.0005_1000_2_0.3_1024_LSTM_0.2_77/model_step_0_release.pt",
                        "models/data_demo_feedback_round2_ced_chren_bpe35000-35000_min10_brnn_adam_0.0005_1000_2_0.3_1024_LSTM_0.2_777/model_step_0_release.pt"]}
chren_opt = _parse_opt(chren_opt)
chren_translator = build_translator(chren_opt, report_score=False)


# moses server
enchr_url = "http://localhost:5001/RPC2"
enchr_proxy = xc.Server(enchr_url)
chren_url = "http://localhost:5002/RPC2"
chren_proxy = xc.Server(chren_url)
# moses QE
import xgboost as xgb
enchr_bst = xgb.Booster({'nthread': 4})
enchr_bst.load_model(f"QE/enchr_demo_xgb.json")
chren_bst = xgb.Booster({'nthread': 4})
chren_bst.load_model(f"QE/chren_demo_xgb.json")


# database
from confidential import MONGO_URI
app.config["MONGO_URI"] = MONGO_URI
mongo = PyMongo(app)


def replace_old_english(word):
    if word in OLD_ENGLISH:
        return OLD_ENGLISH[word]
    else:
        return word


def enchr_translate(src):
    src_tokens = mtr.truecase(' '.join(mtok.tokenize(src)))
    scores, predictions, attns = enchr_translator.translate([' '.join(src_tokens)], batch_size=1, attn_debug=True)
    trg_tokens = predictions[0][0].split(' ')
    pred = md.detokenize(trg_tokens)
    score = np.exp(float(scores[0][0]) / (len(trg_tokens) + 1e-12)) * 5
    # word alignment
    word_alignment = []
    for i, trg_token in enumerate(trg_tokens):
        for j, src_token in enumerate(src_tokens):
            word_alignment.append([f"{j} {src_token}", f"{i} {trg_token}", round(attns[0][i][j], 2)])
    width, height = 30 * len(src_tokens), 30 * len(trg_tokens)
    # look up dictionary
    table, table_height = look_up_dictionary(src_tokens, [])
    return pred, score, word_alignment, width, height, table, table_height


def chren_translate(src):
    def _merge_attn_source(bpe_tokens, attns):
        new_attns, pre_attn = [], None
        for i in range(len(bpe_tokens)):
            if "@@" in bpe_tokens[i]:
                if pre_attn is not None:
                    pre_attn += attns[i]
                else:
                    pre_attn = attns[i]
            else:
                if pre_attn is not None:
                    pre_attn += attns[i]
                    new_attns.append(pre_attn)
                    pre_attn = None
                else:
                    new_attns.append(attns[i])
        return new_attns

    def _merge_attn_target(bpe_tokens, attns):
        new_attns = []
        pre_attn, count = None, 0
        for i in range(len(bpe_tokens)):
            if "@@" in bpe_tokens[i]:
                if pre_attn is not None:
                    pre_attn += attns[i]
                else:
                    pre_attn = attns[i]
                count += 1
            else:
                if pre_attn is not None:
                    pre_attn += attns[i]
                    count += 1
                    pre_attn /= count
                    new_attns.append(pre_attn)
                    pre_attn, count = None, 0
                else:
                    new_attns.append(attns[i])
        if pre_attn is not None:
            new_attns.append(pre_attn / count)
        return new_attns

    src_tokens = mtok.tokenize(src)
    src = bpe.process_line(' '.join(src_tokens))
    src_bpe_tokens = src.split(' ')
    scores, predictions, attns = chren_translator.translate([src], batch_size=1, attn_debug=True)
    pred = predictions[0][0]
    trg_bpe_tokens = pred.split(' ')
    pred = pred.replace("@@ ", "")
    trg_tokens = mdtr.detruecase(pred)
    trg_tokens = [replace_old_english(word) for word in trg_tokens]  # replace Old English tokens
    pred = md.detokenize(trg_tokens)
    score = np.exp(float(scores[0][0]) / (len(trg_bpe_tokens) + 1e-12)) * 5
    # merge attns
    new_attns = np.array(_merge_attn_target(trg_bpe_tokens, np.array(attns[0])))
    new_attns = _merge_attn_source(src_bpe_tokens, np.transpose(new_attns))
    # word alignment
    word_alignment = []
    for i, trg_token in enumerate(trg_tokens):
        for j, src_token in enumerate(src_tokens):
            word_alignment.append([f"{j} {src_token}", f"{i} {trg_token}", round(new_attns[j][i], 2)])
    width, height = 30 * len(src_tokens), 30 * len(trg_tokens)
    # look up dictionary
    table, table_height = look_up_dictionary([], src_tokens)
    return pred, score, word_alignment, width, height, table, table_height


def smt_enchr_translate(src):
    src_tokens = mtr.truecase(' '.join(mtok.tokenize(src)))
    params = {"text": ' '.join(src_tokens), "nbest": 1, "word-align": "true", "add-score-breakdown": "true"}
    result = enchr_proxy.translate(params)
    result = result["nbest"][0]
    trg_tokens = result["hyp"].strip().split(' ')
    pred = md.detokenize(trg_tokens)
    # QE
    scores = result["scores"]
    x = [len(trg_tokens), result["totalScore"]]
    for key in ["Distortion0", "LM0", "LexicalReordering0", "PhrasePenalty0", "TranslationModel0", "WordPenalty0"]:
        x.extend(scores[key][0])
    x += [v / x[0] for v in x[1:]]
    x = xgb.DMatrix(np.array([x]))
    score = enchr_bst.predict(x)[0] / 20
    # word alignment
    word_alignment = []
    for i, trg_token in enumerate(trg_tokens):
        for j, src_token in enumerate(src_tokens):
            word_alignment.append([f"{j} {src_token}", f"{i} {trg_token}", 0.0])
    for alignment in result["word-align"]:
        trg_token_index = alignment["target-word"]
        src_token_index = alignment["source-word"]
        word_alignment[trg_token_index * len(src_tokens) + src_token_index][2] = 1.0
    width, height = 30 * len(src_tokens), 30 * len(trg_tokens)
    # look up dictionary
    table, table_height = look_up_dictionary(src_tokens, [])
    return pred, score, word_alignment, width, height, table, table_height


def smt_chren_translate(src):
    src_tokens = mtok.tokenize(src)
    params = {"text": ' '.join(src_tokens), "nbest": 1, "word-align": "true", "add-score-breakdown": "true"}
    result = chren_proxy.translate(params)
    result = result["nbest"][0]
    pred = result["hyp"].strip()
    trg_tokens = mdtr.detruecase(pred)
    trg_tokens = [replace_old_english(word) for word in trg_tokens]  # replace Old English tokens
    pred = md.detokenize(trg_tokens)
    # QE
    scores = result["scores"]
    x = [len(trg_tokens), result["totalScore"]]
    for key in ["Distortion0", "LM0", "LexicalReordering0", "PhrasePenalty0", "TranslationModel0", "WordPenalty0"]:
        x.extend(scores[key][0])
    x += [v / x[0] for v in x[1:]]
    x = xgb.DMatrix(np.array([x]))
    score = chren_bst.predict(x)[0] / 20
    # word alignment
    word_alignment = []
    for i, trg_token in enumerate(trg_tokens):
        for j, src_token in enumerate(src_tokens):
            word_alignment.append([f"{j} {src_token}", f"{i} {trg_token}", 0.0])
    for alignment in result["word-align"]:
        trg_token_index = alignment["target-word"]
        src_token_index = alignment["source-word"]
        word_alignment[trg_token_index * len(src_tokens) + src_token_index][2] = 1.0
    width, height = 30 * len(src_tokens), 30 * len(trg_tokens)
    # look up dictionary
    table, table_height = look_up_dictionary([], src_tokens)
    return pred, score, word_alignment, width, height, table, table_height


def look_up_dictionary(en_tokens, chr_tokens):
    terms = []
    existing_tokens = set()
    existing_terms = set()
    for token, chr in zip(en_tokens + chr_tokens, [False] * len(en_tokens) + [True] * len(chr_tokens)):
        if len(existing_terms) >= 100 or len(terms) >= MAX_TERMS:
            break
        if token.lower() in existing_tokens or token in punctuations or (not chr and token in stopwords):
            continue
        existing_tokens.add(token.lower())
        try:
            if chr:
                res = requests.get(url=f"https://cherokeedictionary.net/jsonsearch/syll/{token}")
            else:
                res = requests.get(url=f"https://cherokeedictionary.net/jsonsearch/en/{token}")
            res = res.json()
        except:
            continue
        for item in res[:50]:  # only go through the top 50
            if len(terms) >= MAX_TERMS:
                break
            if item["syllabaryb"] == "":
                continue
            chr_token = item["syllabaryb"]
            chr_translit = item["entrytranslit"] if item["entrytranslit"] else item["entrya"]
            en_token = item["definitiond"]
            if chr_token in existing_terms:
                continue
            existing_terms.add(chr_token)
            if chr:
                chr_token_set = chr_token.split(' ')
                if token not in chr_token_set:
                    continue
            else:
                en_token_set = en_token.lower().split(' ')
                if token.lower() not in en_token_set:
                    continue
            chr_sentence, chr_translit_sentence, en_sentence = '', '', ''
            if "sentenceq" in item and item["sentenceq"] is not None:
                chr_sentence = item["sentenceq"].replace("//", "/")
                chr_translit_sentence = item["sentencetranslit"].replace("//", "/")
                en_sentence = item["sentenceenglishs"].replace("//", "/")
            term = f'<tr> <td>{chr_token}<br>{chr_translit}</td> <td>{en_token}</td> <td>{chr_sentence}<br>{chr_translit_sentence}<br>{en_sentence}</td> </tr>'
            terms.append(term)
            break
    table_height = 120 * len(terms) + 50
    terms = ' '.join(terms)
    table = f'<table class="table table-striped"> <thead><tr><th scope="col" style="width: 300px">' \
            f'Cherokee Syllabary/Phonetic</th><th scope="col" style="width: 200px">English</th> ' \
            f'<th scope="col">Sentence</th></tr></thead><tbody>{terms}</tbody></table>'
    return table, table_height


@app.route('/toen', methods=['POST'])
def toen():
    en, en_qe = "", 0.0
    word_alignment, width, height = [], 0, 0
    table, table_height = "", 0
    if request.method == "POST":
        chr = request.form.get("chr")
        model = request.form.get("model")
        if model == "nmt":
            if chr.strip() != '':
                en, en_qe, word_alignment, width, height, table, table_height = chren_translate(chr)
                align, dictionary = True, True if table_height > 0 else False
        elif model == "smt":
            if chr.strip() != '':
                en, en_qe, word_alignment, width, height, table, table_height = smt_chren_translate(chr)
                align, dictionary = True, True if table_height > 0 else False
    if width > 0:
        width = min(width, 450) + 100
        height = min(height, 450) + 150
    if table_height > 0:
        table_height = min(table_height, 400)
    return {"en": en, "en_qe": en_qe, "word_alignment": word_alignment, "width": width, "height": height,
            "table": table, "table_height": table_height}


@app.route('/tochr', methods=['POST'])
def tochr():
    chr, chr_qe = "", 0.0
    word_alignment, width, height = [], 0, 0
    table, table_height = "", 0
    if request.method == "POST":
        en = request.form.get("en")
        model = request.form.get("model")
        if model == "nmt":
            if en.strip() != '':
                chr, chr_qe, word_alignment, width, height, table, table_height = enchr_translate(en)
        elif model == "smt":
            if en.strip() != '':
                chr, chr_qe, word_alignment, width, height, table, table_height = smt_enchr_translate(en)
    if width > 0:
        width = min(width, 450) + 100
        height = min(height, 450) + 150
    if table_height > 0:
        table_height = min(table_height, 400)
    return {"chr": chr, "chr_qe": chr_qe, "word_alignment": word_alignment, "width": width, "height": height,
            "table": table, "table_height": table_height}


@app.route('/', methods=['GET', 'POST'])
def index():
    # get examples
    chrs, chrs_id = [], []
    cursors = mongo.db.chr.find({"status": "unlabeled"}).limit(5)
    for cursor in cursors:
        chrs.append(cursor["text"])
        chrs_id.append(cursor["uid"])
    ens, ens_id = [], []
    cursors = mongo.db.en.find({"status": "unlabeled"}).limit(5)
    for cursor in cursors:
        ens.append(cursor["text"])
        ens_id.append(cursor["uid"])
    en, chr, nmt, tochr, toen = "", "", True, False, False
    en_qe, chr_qe = 0.0, 0.0
    align, word_alignment, width, height = False, [], 0, 0
    dictionary, table, table_height = False, "", 0
    return render_template('index.html', en=en, chr=chr, tochr=tochr, toen=toen,
                           nmt=nmt, en_qe=en_qe, chr_qe=chr_qe,
                           align=align, word_alignment=word_alignment, width=width, height=height,
                           dictionary=dictionary, table=table, table_height=table_height,
                           chrs=chrs, chrs_id=chrs_id, ens=ens, ens_id=ens_id)


@app.route('/expert', methods=['GET', 'POST'])
def expert():
    en, chr, nmt, tochr, toen = "", "", True, False, False
    en_qe, chr_qe = 0.0, 0.0
    align, word_alignment, width, height = False, [], 0, 0
    dictionary, table, table_height = False, "", 0
    if request.method == "POST":
        if request.form["model"] == "nmt":
            nmt = True
            if request.form["action"] == "tochr":
                tochr, toen = True, False
                en = request.form["en"]
                if en.strip() != '':
                    chr, chr_qe, word_alignment, width, height, table, table_height = enchr_translate(en)
                    align, dictionary = True, True if table_height > 0 else False
            elif request.form["action"] == "toen":
                toen, tochr = True, False
                chr = request.form["chr"]
                if chr.strip() != '':
                    en, en_qe, word_alignment, width, height, table, table_height = chren_translate(chr)
                    align, dictionary = True, True if table_height > 0 else False
        elif request.form["model"] == "smt":
            nmt = False
            if request.form["action"] == "tochr":
                tochr, toen = True, False
                en = request.form["en"]
                if en.strip() != '':
                    chr, chr_qe, word_alignment, width, height, table, table_height = smt_enchr_translate(en)
                    align, dictionary = True, True if table_height > 0 else False
            elif request.form["action"] == "toen":
                toen, tochr = True, False
                chr = request.form["chr"]
                if chr.strip() != '':
                    en, en_qe, word_alignment, width, height, table, table_height = smt_chren_translate(chr)
                    align, dictionary = True, True if table_height > 0 else False
    if width > 0:
        width = min(width, 450) + 100
        height = min(height, 450) + 150
    if table_height > 0:
        table_height = min(table_height, 400)
    return render_template('expert.html', en=en, chr=chr, tochr=tochr, toen=toen,
                           nmt=nmt, en_qe=en_qe, chr_qe=chr_qe,
                           align=align, word_alignment=word_alignment, width=width, height=height,
                           dictionary=dictionary, table=table, table_height=table_height)


@app.route('/expertfeedback', methods=['GET', 'POST'])
def expertfeedback():
    if request.method == "POST":
        rate = request.form.get("rate")
        type = request.form.get("type")
        text = request.form.get("text")
        en = request.form.get("en")
        chr = request.form.get("chr")
        model = request.form.get("model")
        comment = request.form.get("comment")
        qe = request.form.get("qe")
        mongo.db.expert.insert_one({"type": type, "model": model, "en": en, "chr": chr, "qe": qe,
                                    "rate": rate, "text": text, "timestamp": time.time(),
                                    "comment": comment})
    return render_template('expert.html')


@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == "POST":
        rate = request.form.get("rate")
        type = request.form.get("type")
        en = request.form.get("en")
        chr = request.form.get("chr")
        model = request.form.get("model")
        comment = request.form.get("comment")
        qe = request.form.get("qe")
        mongo.db.user.insert_one({"type": type, "model": model, "en": en, "chr": chr, "qe": qe,
                                  "rate": rate, "timestamp": time.time(),
                                  "comment": comment})
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=False)