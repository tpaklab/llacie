import re

# A version of this was incorporated into ShortHPISectionRegexStrategy
# It uses the circumstance that EDW replaced newlines with two spaces to
# reformat note text into a more readable form, by hardwrapping it

def fix_note_whitespace(text):
    text = re.sub("  ", "\n", text)
    text = re.sub("\n[ ?]+", "\n", text)
    text = re.sub("\n\n+", "\n\n", text)
    return text.strip()

##################################################################
# Functions for cleaning up list-like text spit out by LLMs
##################################################################

def split_listlike_text(raw_text):
    numeric_list = re.match(r'[1]([).]) +.+?([;,]?) [2]\1( +.+?(\2 |\2? and )\d+\1)*', raw_text)
    bulleted_list = re.match(r'([·•*-] ?)[^\n]+(\n\n?)\1\S', raw_text)

    if raw_text.startswith('\\begin{itemize}'):
        # LLaMA sometimes composes in LaTeX-style lists
        without_prefix = re.sub(r'^\\begin\{itemize\}\s+\\item +', '', raw_text)
        middle = re.sub(r'\s+\\end\{itemize\}[\s\S]*', '', without_prefix)
        values = re.split(r'\s+\\item +', middle)
    elif numeric_list is not None:
        without_suffix = raw_text
        # depending on the style of list, we look for the end of the sentence or paragraph
        end_regex = r'([.]\s+|\n\n)' if numeric_list[1] == ')' else r'[.]?\n\n'
        end_regex_match = re.search(end_regex, raw_text[numeric_list.span()[1]:])
        if end_regex_match is not None:
            without_suffix = raw_text[:numeric_list.span()[1] + end_regex_match.span()[0]]
        middle = re.sub(r'^\d[' + numeric_list[1] + r']', '', without_suffix)
        if numeric_list[2] != '':
            values = re.split(r'(?:' + numeric_list[2] + r'|' + numeric_list[2] + r'? and)? \d+[' + 
                    numeric_list[1] + r']', middle)
        else:
            values = re.split(r'(?: and)? \d+[' + numeric_list[1] + r']', middle)
    elif bulleted_list is not None:
        bullet_regex = r'[' + bulleted_list[1][0] + r']' + bulleted_list[1][1:]
        without_prefix = re.sub(r'^' + bullet_regex, '', raw_text)
        middle = re.sub(r'\n\n[^' + bulleted_list[1][0] + r'][\s\S]*', '', without_prefix)
        values = re.split(bulleted_list[2] + bullet_regex, middle)
    else:
        # Try treating it like a list within a sentence (fragment)
        # Also, excise any superfluous first bullets, numbers, etc
        first_sentence = re.sub(r'^(1[.)]|[·•*-])\s+', '', raw_text)
        first_sentence = re.sub(r'([.]\s+|[.]?\n\n)[\s\S]*', '', first_sentence)
        # If it's not plausibly an inline list, with at least three items, just abort
        if re.search(r'([;,]) \S+.*(\1|\1? and) \S', first_sentence) is None: return None 
        values = re.split(r'[;,] (?:and )?|[;,]? and ', first_sentence)

    return values

def excise_parentheticals(values):
    values = [re.sub(r'\s*[(][^)]+[)]', '', val) for val in values]
    return values

def resplit_list_values(values):
    values = filter(lambda val: re.match(r'No\s+', val, re.IGNORECASE) is None, values)
    values = [re.split(r'[;,] (?:and )?|[;,]? and | */ *(?=[a-zA-Z]{2})|\n', val) for val in values]
    values = [val.strip(' -.') for sublist in values for val in sublist]
    return values

def is_acceptable_value(value):
    if re.match(r'No\s+', value, re.IGNORECASE) is not None: return False
    if re.search(r'[a-z]', value, re.IGNORECASE) is None: return False
    return True

def cleanup_presenting_sx(llm_output_raw):
    values = split_listlike_text(llm_output_raw)
    if values is None: return None
    values = excise_parentheticals(values)
    values = resplit_list_values(values)
    # TODO: excise timing qualifiers (e.g., X days of, for X days, x N days)
    values = filter(is_acceptable_value, values)
    return "\n".join(values)