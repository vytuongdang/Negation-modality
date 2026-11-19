"""String matcher: does an answer mention a component? Shared by the baseline filter and later scoring."""
import re


def _expand(forms):
    """Add simple plurals so 'puppy' also matches 'puppies'."""
    out = set()
    for f in forms:
        if not f:
            continue
        out |= {f, f + "s", f + "es"}
        if re.search(r"[^aeiou]y$", f):
            out.add(f[:-1] + "ies")
    return out


def _alt(forms):
    forms = sorted({f for f in forms if f}, key=len, reverse=True)
    return "|".join(re.escape(f) for f in forms)


def _strip_longer(text, form_list, others):
    """Blank out other components' longer phrases that contain one of our forms (e.g. 'hot dog' when looking for 'dog')."""
    longer = [o for c in others for o in c.get("aliases", []) + c.get("subject_aliases", [])
              if any(f != o and re.search(rf"\b{re.escape(f)}\b", o) for f in form_list)]
    for o in sorted(set(longer), key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(o)}\b", " ", text)
    return text


# adjective derivations accepted for attribute values: metal -> metallic, red -> reddish
ATTR_SUFFIX = r"(?:s|es|ic|lic|ish|dish|y)?"

# Extra names / dropped aliases for existence objects (missed or false hits seen in the H1 pilot).
EXTRA_ALIASES = {"dining table": ["table"], "tv": ["monitor"], "couch": ["sofa"], "cell phone": ["phone", "smartphone"],
                 "potted plant": ["plant"], "sports ball": ["ball"], "wine glass": ["glass"]}
DROP_ALIASES = {"laptop": ["computer"]}  # "desktop computer" is not a laptop
# How models describe CLEVR / VAW values in words outside the annotation vocabulary.
VALUE_SYNONYMS = {"rubber": ["matte"], "metal": ["shiny", "metallic"], "wooden": ["wood"], "golden": ["gold"],
                  "woolen": ["wool"], "large": ["big"], "small": ["little"]}
# Words that turn an attribute alias into a phrase ("big size", "green tone"); the bare word is the alias.
SLOT_WORDS = {"color", "colour", "shade", "tone", "hue", "tint", "size", "stature", "build", "material", "type",
              "surface", "appearance", "look", "finish", "texture", "shape", "substance", "composition", "construction"}
# "a fork" states one fork in an answer to a count question, unless a quantifier follows ("a couple of forks").
A_NOT_ONE = r"(?!\s+(?:couple|few|pair|lot|lots|bunch|number|group|set|pile|stack|dozen|handful|variety|range)\b)"


def _existence_forms(comp):
    forms = [f for f in comp["aliases"] if f not in DROP_ALIASES.get(comp["subject"], [])]
    return forms + EXTRA_ALIASES.get(comp["subject"], [])


def _attribute_forms(comp):
    forms = set(comp["aliases"])
    subj = comp.get("subject", "")
    for a in comp["aliases"]:
        words = a.split()
        if len(words) == 2 and (words[1] in SLOT_WORDS or words[1] == subj):
            forms.add(words[0])
    for v in [comp.get("value")] + list(forms):
        forms |= set(VALUE_SYNONYMS.get(v, []))
    return sorted(f for f in forms if f)


def mentions(answer, comp, others=()):
    text = re.sub(r"[*_`#]+", " ", answer.lower())  # markdown such as "**Car**: 1"
    if comp["type"] == "count":
        nums = _alt(comp["aliases"])
        if comp.get("value") == 1:
            nums = rf"(?:{nums})|(?:an?{A_NOT_ONE})"
        subj_forms = comp.get("subject_aliases") or [comp["subject"]]
        text = _strip_longer(text, subj_forms, others)
        subj = _alt(_expand(subj_forms))
        # "2 dogs", "two large dogs", "two of the dogs" | "dogs: 2", "dogs - two", "dogs (2)", "dogs x2"
        gap = r"(?:\s+(?!and\b|or\b|but\b)[a-z-]+){0,2}?"
        before = rf"\b(?:{nums})\b{gap}\s+(?:{subj})\b"
        after = rf"\b(?:{subj})\b\s*(?:[:=(\-–]|x)\s*(?:{nums})\b"
        return bool(re.search(before, text) or re.search(after, text))
    forms = _attribute_forms(comp) if comp["type"] == "attribute" else (
        _existence_forms(comp) if comp["type"] == "existence" else comp["aliases"])
    text = _strip_longer(text, forms, others)
    if comp["type"] == "attribute":
        return bool(re.search(rf"\b(?:{_alt(forms)}){ATTR_SUFFIX}\b", text))
    return bool(re.search(rf"\b(?:{_alt(_expand(forms))})\b", text))
