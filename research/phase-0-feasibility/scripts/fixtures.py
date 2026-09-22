"""Parity fixtures: length-controlled synthetic requests plus hand-written edge cases.

All text here was written for this project (no third-party fixture text is copied).
"""

from __future__ import annotations

from common import QUESTION_POOL, agent_config, checkpoint, make_request, valid_lengths

SEEDS = (11, 12, 13)
QUESTIONS_PER_CASE = 8  # cycles through all eight pool questions: 3 noul, 2 choice, 2 score, 1 noul

MULTILINGUAL = {
    "en": "My card was charged two times for order 5521. I want the second charge refunded.",
    "de": "Meine Karte wurde für Bestellung 5521 zweimal belastet. Bitte erstatten Sie die zweite Zahlung.",
    "fr": "Ma carte a été débitée deux fois pour la commande 5521. Merci de rembourser le second débit.",
    "es": "Me cobraron dos veces el pedido 5521. Quiero que me devuelvan el segundo cargo.",
    "zh": "订单5521被扣了两次款，请把第二笔退给我。",
    "ja": "注文5521で二回請求されました。二回目の請求を返金してください。",
    "hi": "ऑर्डर 5521 के लिए मेरे कार्ड से दो बार पैसे कटे। कृपया दूसरा भुगतान वापस करें।",
    "ru": "За заказ 5521 с карты списали деньги дважды. Верните, пожалуйста, второй платёж.",
    "ar": "تم خصم المبلغ مرتين للطلب 5521. أرجو استرداد الخصم الثاني.",
}

EDGE_QUESTIONS = {
    "structured_choice": {
        "type": "choice",
        "instructions": {"task": "route the ticket", "channel": "email"},
        "criteria": {"billing": {"covers": ["refunds", "invoices"]}, "other": False},
    },
    "five_level_score": {
        "type": "score",
        "instructions": "Rate the customer's frustration.",
        "criteria": ["calm", "mildly annoyed", "annoyed", "angry", "furious"],
    },
    "noul_custom": {
        "type": "noul",
        "instructions": "Is money owed back to the customer?",
        "criteria": {"true": "a refund or credit is due", "false": "nothing is owed"},
    },
    "twenty_options": {
        "type": "choice",
        "instructions": "Which queue should receive this?",
        "criteria": ["billing"] + [f"queue_{i:02d}" for i in range(19)],
    },
}


def synthetic_cases(model: str):
    from laya_coreml.tokenizer import Tokenizer

    tok = Tokenizer(checkpoint(model) / "tokenizer")
    cfg = agent_config(model)
    cases = []
    for L in valid_lengths(model):
        for seed in SEEDS:
            state, questions, _ = make_request(tok, cfg, L, QUESTIONS_PER_CASE, seed=seed)
            cases.append({"name": f"L{L}-s{seed}", "length": L, "state": state, "questions": questions})
    return cases


def edge_cases():
    base_q = {f"q{i}": QUESTION_POOL[i] for i in range(3)}
    cases = []
    for lang, text in MULTILINGUAL.items():
        cases.append({"name": f"lang-{lang}", "state": {"message": text}, "questions": base_q})
    cases += [
        {"name": "empty-state", "state": "", "questions": base_q},
        {
            "name": "conversation",
            "state": [
                {"role": "user", "content": MULTILINGUAL["en"]},
                {"role": "agent", "content": "Sorry about that, checking now."},
            ],
            "questions": base_q,
        },
        {"name": "mask-literals", "state": "[MASK] <mask> the [MASK] refund <mask>", "questions": base_q},
        {"name": "edge-questions", "state": {"message": MULTILINGUAL["en"]}, "questions": EDGE_QUESTIONS},
        {
            "name": "overflow-truncated",
            "state": " ".join(["The duplicate charge still has not been refunded."] * 400),
            "questions": base_q,
        },
    ]
    return cases


def all_cases(model: str):
    return synthetic_cases(model) + edge_cases()
