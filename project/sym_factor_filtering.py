
factors_list = np.array([])
for row in expr_matrix:
    for factor in row.as_ordered_terms():
        factors_list = np.append(factors_list, list(factor.as_coeff_mul()[-1]) )

factors_dict = {}
for factor in factors_list:
    if factor in factors_dict.keys():
        factors_dict[factor] += 1
    else:
        factors_dict[factor] = 1

def decompose_factors_dict(factors_dict):
    '''Take the dictionary of factors in the impact equations, and breaks
    them down further. This process can be repeated to get the factors only
    in terms of sines, cosines, numbers, and symbolic variables.
    
    Returns: new_factors_dict. Contains the same data as factors_dict
        in smaller terms.
    '''
    new_factors_array = np.array([])
    new_factors_dict = factors_dict.copy()

    for factor in factors_dict.keys():
        if factor.is_Add:
            #add components to list of factors and remove from old dictionary
            new_factors_array = np.append(new_factors_array, factor.as_ordered_terms())
            del new_factors_dict[factor]

        if factor.is_Pow:        
            new_factors_array = np.append(new_factors_array, list(factor.as_powers_dict().keys()))
            del new_factors_dict[factor]

        if factor.is_Mul:
            new_factors_array = np.append(new_factors_array, list(factor.as_coeff_mul()[-1]) )
            del new_factors_dict[factor]

    #fdo data checking and add terms back into the dictionary
    for factor in new_factors_array:
        if factor in new_factors_dict.keys():
            new_factors_dict[factor] += 1
        else:
            new_factors_dict[factor] = 1
            
    return new_factors_dict

new_factors_dict = decompose_factors_dict(factors_dict)
factors_dict_v3 = decompose_factors_dict(new_factors_dict)
factors_dict_v4 = decompose_factors_dict(factors_dict_v3)
factors_list_v5 = list(factors_dict_v4.keys())
factors_list_v6 = [f for f in factors_list_v5 if type(f) == sym.cos or type(f) == sym.sin]

