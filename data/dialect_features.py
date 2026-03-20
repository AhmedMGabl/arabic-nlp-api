"""
Arabic Dialect Fingerprints
Character n-gram and word-level features for dialect detection.
Each dialect is represented by high-frequency patterns that distinguish it
from other varieties of Arabic.

Dialects covered:
  MSA   - Modern Standard Arabic (الفصحى)
  EGY   - Egyptian (المصري)
  GULF  - Gulf / Arabian Peninsula (الخليجي)
  LEV   - Levantine — Syria / Lebanon / Jordan / Palestine (الشامي)
  MAG   - Maghrebi — Morocco / Algeria / Tunisia / Libya (المغربي)
"""

from typing import NamedTuple


class DialectProfile(NamedTuple):
    code: str
    name_en: str
    name_ar: str
    keywords: list[str]      # High-signal words/particles
    bigrams: list[str]       # High-signal character bigrams  (Arabic script)
    trigrams: list[str]      # High-signal character trigrams
    weight: float            # Prior weight (rough corpus frequency)


DIALECT_PROFILES: list[DialectProfile] = [
    DialectProfile(
        code="MSA",
        name_en="Modern Standard Arabic",
        name_ar="العربية الفصحى",
        keywords=[
            "الذي", "التي", "الذين", "اللاتي",
            "إن", "أن", "لكن", "حيث", "إذا",
            "منذ", "حتى", "سوف", "قد", "لقد",
            "يجب", "ينبغي", "يمكن", "يستطيع",
            "وفقاً", "بناءً", "نظراً", "علاوة",
            "بالإضافة", "وبالتالي", "وبذلك",
            "كما", "أيضاً", "أيضا", "بينما",
            "ومع", "وقد", "وكان", "وأن",
        ],
        bigrams=["وا", "ال", "وف", "قد", "يو", "هذ", "ذل"],
        trigrams=["الذ", "لذي", "وفق", "بنا", "يجب", "كان", "هذا"],
        weight=0.20,
    ),
    DialectProfile(
        code="EGY",
        name_en="Egyptian",
        name_ar="العامية المصرية",
        keywords=[
            "عشان", "بتاع", "بتاعة", "بتوع",
            "إيه", "ايه", "فين", "منين",
            "ازاي", "إزاي", "ليه", "إيه",
            "كده", "كدا", "مش", "دلوقتي",
            "هيجي", "جاي", "جايي", "روح",
            "خليني", "خلي", "معلش", "طب",
            "بقى", "بقا", "أهو", "زي",
            "أنا", "إنت", "هو", "هي",
            "احنا", "إحنا", "انتوا", "هما",
            "أوي", "خالص", "كتير",
            "عندي", "عندك", "عنده",
            "مفيش", "فيه", "فيها",
        ],
        bigrams=["عش", "شا", "بت", "اع", "ده", "كد", "مش"],
        trigrams=["عشا", "شان", "بتا", "تاع", "كده", "مش "],
        weight=0.25,
    ),
    DialectProfile(
        code="GULF",
        name_en="Gulf Arabic",
        name_ar="العامية الخليجية",
        keywords=[
            "وايد", "زين", "شلونك", "شخبارك",
            "ليش", "وين", "شنو", "ما أدري",
            "كيفك", "كيف", "يبي", "ابي",
            "حق", "مالت", "مالك", "مالي",
            "بس", "هاو", "هو", "هي",
            "نبي", "أبغى", "أبي", "ودي",
            "يالله", "يله", "لحظة",
            "عيل", "عيال", "الحين",
            "احين", "دحين", "الله",
            "مافي", "فيه",
        ],
        bigrams=["وي", "ين", "شل", "لو", "نك", "حق", "زي"],
        trigrams=["واي", "ايد", "شلو", "لون", "حقك", "زين"],
        weight=0.20,
    ),
    DialectProfile(
        code="LEV",
        name_en="Levantine Arabic",
        name_ar="العامية الشامية",
        keywords=[
            "هيك", "هيدا", "هيدي", "شو",
            "كيفك", "وين", "لوين", "منين",
            "بدي", "بده", "بدك", "بدنا",
            "بدهم", "رح", "ما رح",
            "عم", "عم بـ", "لأ", "لا2",
            "مشان", "مشان هيك",
            "ياي", "حالك", "شو بتحب",
            "تكرم", "يسلمو", "يسلم",
            "هلق", "هلأ", "هلقيت",
            "زاكي", "منيح", "متل",
        ],
        bigrams=["هي", "يك", "شو", "بد", "رح", "عم", "لأ"],
        trigrams=["هيك", "يدا", "شوب", "بدي", "رحا", "عمب"],
        weight=0.20,
    ),
    DialectProfile(
        code="MAG",
        name_en="Maghrebi Arabic",
        name_ar="الدارجة المغاربية",
        keywords=[
            "واش", "كيفاش", "فاش", "علاش",
            "بزاف", "برشا", "يسر",
            "كي", "كيما", "بحال",
            "نتا", "نتي", "هو", "هي",
            "انا", "احنا", "نتوما",
            "عندي", "عندك", "عنده",
            "ماعنديش", "ماكانش",
            "مزيان", "كيف داير",
            "لاباس", "لاباس عليك",
            "رانا", "راني", "راهو",
            "هاك", "هاكا", "ديما",
        ],
        bigrams=["وا", "اش", "بز", "زا", "كي", "مز", "رن"],
        trigrams=["واش", "اشن", "بزا", "زاف", "كيف", "مزي"],
        weight=0.15,
    ),
]

# Build a lookup dict for fast access
DIALECT_BY_CODE: dict[str, DialectProfile] = {d.code: d for d in DIALECT_PROFILES}
