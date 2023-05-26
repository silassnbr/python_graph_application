from rouge import Rouge

def rouge_scores(reference_text, candidate_text):
    rouge = Rouge()
    scores = rouge.get_scores(candidate_text, reference_text)

    # recall, precision, f1
    r1 = scores[0]['rouge-1']
    r2 = scores[0]['rouge-2']
    rl = scores[0]['rouge-l']

    return r1, r2, rl

def rouge_r_or_p(captured, words):

    try:
        rouge_score = len(captured) / len(words)
    except TypeError:
        rouge_score = captured / len(words)    
    except ZeroDivisionError:
        rouge_score = 0    

    return round(rouge_score,4)

def rouge_f(r, p):
    try:
        f = 2.0 * (p * r) / (p + r + 0.00000001)
    except ZeroDivisionError:
        f = 0    
    return round(f,4)

def calculate_rouge_1(reference_text, candidate_text):
    
    reference_words = reference_text.lower().split() 
    candidate_words = candidate_text.lower().split()

    matching_words = []

    for ngram in set(reference_words):
        count_reference_words = reference_words.count(ngram)
        count_candidate_words = candidate_words.count(ngram)
        if count_reference_words > 0 and count_candidate_words > 0:
            matching_words.extend([ngram] * min(count_reference_words, count_candidate_words))
            
    # print(matching_words, len(matching_words))
    # ROUGE-1
    rouge_1_r = rouge_r_or_p(matching_words, reference_words)
    rouge_1_p = rouge_r_or_p(matching_words, candidate_words)
    rouge_1_f = rouge_f(rouge_1_r, rouge_1_p)
    
    return rouge_1_r, rouge_1_p, rouge_1_f


def calculate_rouge_2(reference_text, candidate_text):
    
    reference_words = reference_text.lower().split() 
    reference_pair = list(zip(*[reference_words[i:] for i in range(2)]))


    candidate_words = candidate_text.lower().split()
    candidate_pair = list(zip(*[candidate_words[i:] for i in range(2)]))

    matching_ngrams = set(reference_pair) & set(candidate_pair)
    
    # ROUGE-2 
    rouge_2_r = rouge_r_or_p(matching_ngrams, reference_pair)
    rouge_2_p = rouge_r_or_p(matching_ngrams, candidate_pair)
    rouge_2_f = rouge_f(rouge_2_r, rouge_2_p)    

    return rouge_2_r, rouge_2_p, rouge_2_f


def calculate_rouge_l(reference, candidate):
    reference_words = reference.lower().split()  
    candidate_words = candidate.lower().split() 

    reference_length = len(reference_words) 
    candidate_length = len(candidate_words) 

    lcs_matrix = []
    row = []

    for i in range(reference_length + 1):

        for j in range(candidate_length + 1):
            row.append(0)

        lcs_matrix.append(row)
        row = []

    for i in range(1, reference_length + 1):
        for j in range(1, candidate_length + 1):
            if reference_words[i-1] == candidate_words[j-1]:
                lcs_matrix[i][j] = lcs_matrix[i-1][j-1] + 1
            else:
                lcs_matrix[i][j] = max(lcs_matrix[i-1][j], lcs_matrix[i][j-1]) # en uzun ortaklık güncelle

    lcs_length = lcs_matrix[reference_length][candidate_length] 

    # ROUGE-L
    rouge_l_r = rouge_r_or_p(lcs_length, reference_words)
    rouge_l_p = rouge_r_or_p(lcs_length, candidate_words)
    rouge_l_f = rouge_f(rouge_l_r, rouge_l_p)

    return rouge_l_r, rouge_l_p, rouge_l_f
