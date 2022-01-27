def array_to_string(array: list):
    result = '('
    for i, ele in enumerate(array):
        if i == 0:
            result += str(ele)
        else:
            result += ', ' + str(ele)
    result += ')'
    return result
