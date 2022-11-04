from pyeda.inter import *
import numpy as np

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
                if (not tail == ""):
                    rv = rv[ 0: i] + "+ " + num_string + "*" + tail
                else:
                    rv = rv[ 0: i] + "+ " + num_string 
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
            coeff_string = monomials[i].split("*")
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

# print(p_convertor( "2*low - 3*high + 15 <= -2 + mid" , ["low", "high", "mid"] ))



def bracket_parser(string, recursion_depth):

    rv = {}
    bracket_ct = 0
    str_index = 0
    prop_ct = 0
    rv_string = ""
    for i in range(len(string)):
        ch = string[i]
        rv_string = rv_string + ch
        if (ch == '('):
            if (bracket_ct == 0):
                str_index = i
            bracket_ct = bracket_ct + 1
        elif (ch == ')'):
            bracket_ct = bracket_ct - 1
            if (bracket_ct == 0):
                key = "P" + str(recursion_depth) + str(prop_ct)
                rv.update({ key : string[str_index+1:i] })
                rv_string = rv_string[:len(rv_string)-(i - str_index+1)]
                rv_string = rv_string + key
                prop_ct = prop_ct + 1
    
    rv2 = {}
    for key in rv:
        if '(' in rv[key]:
            T = bracket_parser( rv[key], recursion_depth + 1)
            rv2.update( T[0]  )
            rv_string = rv_string.replace(key.strip(), "(" + T[1] + ")")
        else:
            rv2.update({ key: rv[key] })

    return (rv2, rv_string)

# A = bracket_parser("((2x + 3y = 20 && 2x - 8u <= 50) and (x + y = 20)) \/ (x-y <= 10)", 0) 
# print(A[0], '\n', A[1])


def logical_connective_parser (string, symb, replace_symb, T):
    n = len(symb)
    rv = {}
    rv_string = ""
    str_index = 0
    prop_ct = 0
    i = 0
    while i < len(string):
        curr = string[i:i+n]
        if (curr == symb or i == len(string) - 1):
            key = "Q" + str(T) + str(prop_ct)
            if (curr == symb):
                rv.update({key: string[str_index : i].strip()})
            else:
                rv.update({key: string[str_index : i+1].strip() })
            rv_string = rv_string[: len(rv_string) - (i - str_index) ]
            if (curr == symb):
                rv_string = rv_string + key + " " + replace_symb
            else:
                rv_string = rv_string + key
            str_index = i + n
            prop_ct = prop_ct + 1
            i = i + n
        else:
            rv_string = rv_string + string[i]
            i = i + 1

    return (rv, rv_string)
        
# A = logical_connective_parser( "2*x + 3*y <= 54 && 3*x - 2*y >= 41 && !p0 && 2*y = 8" , "&&" , " /\ " )
# print(A[0], A[1])

def negation(pred):
    temp = pred.copy()
    op = pred[-2]
    if (op > 0):
        temp[-2] = temp[-2] - 3 
        return [ temp ]
    elif (op < 0):
        temp[-2] = temp[-2] + 3 
        return [ temp ]
    else:
        temp2 = temp.copy()
        temp[-2] = -1
        temp2[-2] = 1
        return [temp, temp2]

def pyeda_conj_parser(exp, D):
    if (not "And" in exp): 
        expr = exp.strip()
        if "~" in expr:
            return negation(D[expr[1:]])
        else:
            return [ D[expr] ]
    else:
        temp = exp.strip()
        exp = temp[4: len(temp) - 1]
        pred_list = exp.split(',')
        rv = []
        for pred in pred_list:
            expr = pred.strip()
            if "~" in expr:
                rv = rv + negation( D[expr[1:]] )
            else: 
                rv.append( D[expr]  )
        return rv

# pyeda expr:  Or(Q50, And(Q30, ~Q40), And(Q31, ~Q40))
def pyeda_dnf_parser(exp, D):
    if (not "Or" in exp): 
        return [ pyeda_conj_parser(exp, D) ]
    else:
        temp = exp.strip()
        exp = temp[3: len(temp) - 1]

        conj_list = []

        bracket_ct = 0
        start_ind = 0
        for i in range(len(exp)):
            ch = exp[i]
            if (ch == '('):
                bracket_ct = bracket_ct + 1
            elif (ch == ')'):
                bracket_ct = bracket_ct - 1
            elif (ch == ',' and bracket_ct == 0):
                conj_list.append( exp[start_ind: i].strip() )
                start_ind = i+1
        conj_list.append( exp[start_ind: ].strip())

        rv = []
        for conj in conj_list:
            rv.append( pyeda_conj_parser(conj, D)  )
        return rv    



def DNF_parser(string, var_vector):

    temp = bracket_parser(string, 0)
    
    # print(temp)
    # print(temp[0], temp[1]) ##MOST WEIRD ERROR???? print(temp) is not same output as print(temp[0], temp[1]) as temp[1] has symbol '\\/' instead of '\/'??????

    curr_dict = temp[0]
    rv_string = temp[1]
    rv_dict = {}
    i = 0
    for key in curr_dict:
        temp_string = curr_dict[key]
        A = logical_connective_parser(curr_dict[key] , "&&", " & ", i)
        rv_dict.update( A[0] )
        if ( len(A[0]) > 1):
            rv_string = rv_string.replace( key , "(" + A[1] + ")" )
        else:
            rv_string = rv_string.replace( key ,  A[1]  ) 
        i = i + 1
    rv_string = rv_string.replace( "&&", " & " )


    rv_dict2 = {}
    for key in rv_dict:
        temp_string = rv_dict[key]
        A = logical_connective_parser(rv_dict[key] , "||", " | ", i)
        rv_dict2.update( A[0] )
        if ( len(A[0]) > 1):
            rv_string = rv_string.replace( key , "(" + A[1] + ")" )
        else:
            rv_string = rv_string.replace( key ,  A[1]  )  
        i = i + 1
    rv_string = rv_string.replace( "||", " | ")


    rv_string = rv_string.replace( "!", "~" )

    final_dict = {}
    for key in rv_dict2:
        final_dict.update({ key: p_convertor(rv_dict2[key], var_vector)})

    pyeda_expr = expr(rv_string).to_dnf()

    A = pyeda_dnf_parser(str(pyeda_expr), final_dict)
    

    return A






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

    rv2 = []
    n = len(var_vector)
    for (i,var) in enumerate(var_vector):
        if (var == lhs):
            rv2.append(rv_list)
        else:
            temp = [0]*(n+1)
            temp[i] = 1
            rv2.append(temp)
    temp = [0]*(n+1)
    temp[n] = 1
    rv2.append(temp) 
    return rv2 #Returns 2D list which is transition for that step

# Static Analysis for Transitions Done!
def transitions_converter (transitions, var_vector):
    n = len(var_vector)
    rv = np.identity(n+1, dtype = int)
    for transition in transitions:
        temp = np.array( transition_converter( transition, var_vector ), ndmin = 2)
        rv = np.matmul(temp , rv)
    
    rv_list = list(rv)
    return [list(x) for x in rv_list]

# print(DNF_parser("(!(2*x + 3*y == 20 || 2*x - 8*u <= 50) && !(x + y == 20)) || (x - y <= 10)", ["u", "x", "y"] ) )
# print(transitions_converter( [ "x = y + 2*z + 3" , "z = y + 1", "y = x + 2 + y" ] , ["x", "y", "z"] ))

def variable_def_formatter (s):
    if "int " in s:
        s = s.replace("int ", " ")
    if ";" in s:
        s = s.replace(";", " ")
    rv = s.split(",")
    return [x.strip() for x in rv]

def transition_str_formatter(s):
    s = s.strip()[: -1]
    rv = s.split(";")
    return [x.strip() for x in rv]


def program_converter ( var_string, P, B, T, Q):
    var_vector = variable_def_formatter(var_string)
    T_list = transition_str_formatter(T)
    print("P: ", DNF_parser(P, var_vector) )
    print("B: ", DNF_parser(B, var_vector) )
    print("Q: ", DNF_parser(Q, var_vector) )
    print("T: ", transitions_converter( T_list, var_vector) )
    return

#NEED TO MANUALLY ADD BRACKETS TO DNFs, else ERROR!!!
# LARGE INT is 10000
V = "int lo, mid, hi;"
P = "(lo == 0) && (mid > 0) && (mid < 10000) && (hi == 2*mid) " #  
B = "(mid > 0)"
T = "lo = lo + 1; hi = hi - 1; mid = mid - 1;"
Q = "(lo == hi)" 

program_converter( V, P, B , T , Q  )