
# P ~ a1x1 + a2x2 + ... anxn op b
# In the form given, there may be constants in LHS and coeff in RHS
# What if there is a - instead of a plus?!?
def p_convertor (string, var_vector):
    if ("==" in string):
        op = 0
        opstring = "=="
    elif (">=" in string):
        op = 1
        opstring = ">="
    elif (">" in string):
        op = 2
        opstring = ">"
    elif ("<=" in string):
        op = -1
        opstring = "<="
    elif ("<" in string):
        op = -2    
        opstring = "<"
    else:
        abort("Opstring not found")

    L = string.split(opstring)
    const = int(L[1].strip())
    sum_coeffs = L[0]     

    monomials = sum_coeffs.split("+")

    coeff_dictionary = {}
    for i in range(len(monomials)):
        if ("*" in monomials[i]):
            coeff_string = monomials[i].split("*")
            coeff_dictionary.update({ coeff_string[1].strip() : int(coeff_string[0].strip()) })
        else:
            remaining_string = monomials[i].strip()
            if (remaining_string.isnumeric() ):
                coeff_dictionary.update({ "constant"  : -int(remaining_string) })
            else:
                coeff_dictionary.update({ remaining_string  : 0 })
    
    rv = []
    for var_str in var_vector:
        rv.append( coeff_dictionary[var_str] )
    rv.append(op)
    rv.append(const)

    return rv    

print(p_convertor( "2*low + 3*high <= 2" , ["low", "high"] ))