
def convert_minus (exp):
    exp = exp.strip()
    rv = exp
    plus_stack = 0
    i = 0
    while i < len(rv):
        ch = rv[i]
        if (ch == '+'):
            plus_stack = 1
        elif (ch == ' ' or ch == '-'):
            plus_stack = plus_stack            
        else:
            plus_stack = 0

        if (ch == '-' and plus_stack == 0):
            tail = rv[i+1 : ].strip()
            num_string = "-1"
            if (tail[0].isnumeric() or tail[0] == '-'):
                j = 0 
                while j < len(tail):
                    ch2 = tail[j]
                    if (  not(ch2.isnumeric() or (j == 0 and ch2 == '-')) ):
                        break
                    j = j + 1
                num_string = str(-1 * int(tail[:j]))
                tail = tail[j+1: ] #As tail[j] = *, and we want to avoid double *
            if (i == 0):
                if (tail[0] == ' ' or tail[0] == '+' or tail[0] == '-'):
                    rv = num_string + " " + tail
                    i = i +  len(num_string) 
                else:
                    rv =  num_string + "*" + tail
                    i = i + len(num_string) 
            else:
                rv = rv[ 0: i] + "+ " + num_string + "*" + tail
                i = i + len(num_string) + 1
        i = i + 1

    return rv

# print( convert_minus("-x -  1*y + -3*z + 4*u - -5*t") )

# P ~ a1x1 + a2x2 + ... anxn op b
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
    
    
    
    lhs = convert_minus(L[0])     
    monomials = lhs.split("+")

    lhs_coeff_dictionary = {}
    for i in range(len(monomials)):
        if ("*" in monomials[i]):
            coeff_string = monomials[i].split("*")
            if (coeff_string[1].strip() in lhs_coeff_dictionary):
                lhs_coeff_dictionary[coeff_string[1].strip()] = lhs_coeff_dictionary[coeff_string[1].strip()] + int(coeff_string[0].strip())
            else:
                lhs_coeff_dictionary.update({ coeff_string[1].strip() : int(coeff_string[0].strip()) })
        else:
            remaining_string = monomials[i].strip()
            if (remaining_string.isnumeric() or (remaining_string[0] == '-' and remaining_string[1:].isnumeric() ) ):
                if ("constant" in lhs_coeff_dictionary):
                    lhs_coeff_dictionary["constant"] = lhs_coeff_dictionary["constant"] +  int(remaining_string)
                else:
                    lhs_coeff_dictionary.update({ "constant"  : int(remaining_string) })
            else:
                if (remaining_string in lhs_coeff_dictionary):
                    lhs_coeff_dictionary[remaining_string] = lhs_coeff_dictionary[remaining_string] + 1
                else:
                    lhs_coeff_dictionary.update({ remaining_string  : 1 })
    for var_str in var_vector:
        if not (var_str in lhs_coeff_dictionary):
            lhs_coeff_dictionary.update({var_str: 0})
    if not ("constant" in lhs_coeff_dictionary):
        lhs_coeff_dictionary.update({"constant": 0})

    rhs = convert_minus(L[1])    


    monomials = rhs.split("+")

    rhs_coeff_dictionary = {}
    for i in range(len(monomials)):
        if ("*" in monomials[i]):
            if (coeff_string[1].strip() in rhs_coeff_dictionary):
                rhs_coeff_dictionary[coeff_string[1].strip()] = rhs_coeff_dictionary[coeff_string[1].strip()] + int(coeff_string[0].strip())
            else:
                rhs_coeff_dictionary.update({ coeff_string[1].strip() : int(coeff_string[0].strip()) })
        else:
            remaining_string = monomials[i].strip()
            
            if (remaining_string.isnumeric() or (remaining_string[0] == '-' and remaining_string[1:].isnumeric() )):
                if ("constant" in rhs_coeff_dictionary):
                    rhs_coeff_dictionary["constant"] = rhs_coeff_dictionary["constant"] +  int(remaining_string)
                else:
                    rhs_coeff_dictionary.update({ "constant"  : int(remaining_string) })
            else:
                if (remaining_string in rhs_coeff_dictionary):
                    rhs_coeff_dictionary[remaining_string] = rhs_coeff_dictionary[remaining_string] + 1
                else:
                    rhs_coeff_dictionary.update({ remaining_string  : 1 })
    for var_str in var_vector:
        if not (var_str in rhs_coeff_dictionary):
            rhs_coeff_dictionary.update({var_str: 0})
    if not ("constant" in rhs_coeff_dictionary):
        rhs_coeff_dictionary.update({"constant": 0})


    rv = []
    for var_str in var_vector:
        rv.append( lhs_coeff_dictionary[var_str] - rhs_coeff_dictionary[var_str] )
    rv.append(op)
    rv.append(rhs_coeff_dictionary["constant"] - lhs_coeff_dictionary["constant"])

    return rv    

def wff_converter(string, var_vector):
    return


def transition_converter( transition, var_vector ):
    
    A = transition.split("=")
    lhs = A[0].strip()
    rhs = convert_minus(A[1])

    monomials = rhs.split("+")

    coeff_dictionary = {}
    for i in range(len(monomials)):
        if ("*" in monomials[i]):
            coeff_string = monomials[i].split("*")
            if (coeff_string[1].strip() in coeff_dictionary):
                coeff_dictionary[coeff_string[1].strip()] = coeff_dictionary[coeff_string[1].strip()] + int(coeff_string[0].strip())
            else:
                coeff_dictionary.update({ coeff_string[1].strip() : int(coeff_string[0].strip()) })
        else:
            remaining_string = monomials[i].strip()
            if (remaining_string.isnumeric() or (remaining_string[0] == '-' and remaining_string[1:].isnumeric() ) ):
                if ("constant" in coeff_dictionary):
                    coeff_dictionary["constant"] = coeff_dictionary["constant"] +  int(remaining_string)
                else:
                    coeff_dictionary.update({ "constant"  : int(remaining_string) })
            else:
                if (remaining_string in coeff_dictionary):
                    coeff_dictionary[remaining_string] = coeff_dictionary[remaining_string] + 1
                else:
                    coeff_dictionary.update({ remaining_string  : 1 })
    for var_str in var_vector:
        if not (var_str in coeff_dictionary):
            coeff_dictionary.update({var_str: 0})
    if not ("constant" in coeff_dictionary):
        coeff_dictionary.update({"constant": 0})


    rv_list = []
    for var_str in var_vector:
        rv_list.append( coeff_dictionary[var_str]  )
    rv_list.append(coeff_dictionary["constant"] )

    return (lhs , rv_list)

# No static Analysis done
def transitions_converter (transitions, var_vector):
    n = len(var_vector)
    transition_dict = {}
    for transition in transitions:
        (x, L) = transition_converter( transition, var_vector )
        transition_dict.update({ x : L  } )
    
    
    for var_str in var_vector:
        if not (var_str in transition_dict):
            val = [0] * n
            val[i] = 1
            transition_dict.update({ var_str : val })

    rv = []
    for (i,var_str) in enumerate(var_vector):

        rv.append( transition_dict[var_str]  )

    return rv





# print(p_convertor( "2*low - 3*high + 15 <= -2 + mid" , ["low", "high", "mid"] ))

# print(transitions_converter( [ "x = y + 2*z + 3" , "z = y + 1", "y = x + 2 + y" ] , ["x", "y", "z"] ))