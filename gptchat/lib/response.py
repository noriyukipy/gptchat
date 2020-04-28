def extract_response_tokens(tokens, start_token, end_token):
    """
    Args:
        tokens (List[str]): tokens consisting of context and response.
            response starts with start_token, and possibly ends with end_token
        start_token (str): Token from which
    """
    start_idx = tokens.index(start_token)
    try:
        end_idx = tokens.index(end_token)
    except ValueError:
        end_idx = len(tokens) - 1
    return tokens[start_idx+1:end_idx]
