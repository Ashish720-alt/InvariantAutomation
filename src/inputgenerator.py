from pyeda.inter import *
import numpy as np


#Can't handle following: '!=' operator, '->' or '<->' logical connective.

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
                if tail == "":
                    rv = num_string
                    i = i + len(num_string)
                elif (tail[0] == ' ' or tail[0] == '+' or tail[0] == '-'):
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

    if string == "":
        return []


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


def program_converter ( var_string, P, B, Ts, Q):
    var_vector = variable_def_formatter(var_string)
    print("Variables: ", var_vector)
    print("P: ", DNF_parser(P, var_vector) )
    print("B: ", DNF_parser(B, var_vector) )
    print("Q: ", DNF_parser(Q, var_vector) )
    print("T: ")
    for (i,x) in enumerate(Ts):
        Bx = x[1]
        Ts_x = x[0]
        if (Bx == ""):
            print("\t: ", end = '')
        else:
            print(DNF_parser(Bx, var_vector) , ": " , end = '')
        print("[ ", end = '')
        for (j,T) in enumerate(Ts_x):
            T_list = transition_str_formatter(T)
            print("(" , transitions_converter( T_list, var_vector), '), ', end = '' )
        print(" ]")
    print("n = ", len(var_vector))
    return

def class_wrapper_program_converter (obj):
    return program_converter( obj.V, obj.P, obj.B , obj.T , obj.Q  )

#NEED TO MANUALLY ADD BRACKETS TO DNFs, else ERROR!!!
# LARGE INT is 1000000
# Check which loop invariant clause system

class loop_lit:
    #  Double loops with  sequential inner loops (total 3 loops)
    class cggmp2005b:  
        A = []

    # Has modulo operator within loop transition as guard of if-else.
    class ddlm2013:
        A = []

    # Has 11 goto operators (didnt check for nesting)
    class gj2007_c_i_plhb_reducer:
        A = []

    # Has 69 goto operators (didnt check for nesting)
    class gj2007_c_i_pnlh_reducer:
        A = []

    class cggmp2005_variant:
        V = "int lo, mid, hi;"
        P = "(lo == 0) && (mid > 0) && (mid < 1000000) && (hi == 2*mid) "  
        B = "(mid > 0)"
        T = [( ["lo = lo + 1; hi = hi - 1; mid = mid - 1;"], "") ]
        Q = "(lo == hi)" 

    class css2003:
        V = "int i,j,k;"
        P = "(i == 1) && (j == 1) && (k >= 0) && (k <= 1)"
        B = "(i < 1000000)"
        T = [ (["i = i + 1; j = j + k; k = k - 1;"], "") ]
        M = "(1 <= i + k) && (i + k <= 2) && (i >= 1)" #This is actually M

    #Conjunctive Randomized loop guard!
    class gcnr2008:
        V = "int x,y,z,w;"
        P = "(x == 0) && (y == 0) && (z == 0) && (w == 0)"
        B = "(y < 10000)"
        T = [ ( ["x = x + 1; y = y + 100; w = w + 1; z = z + 10;" , "x = x + 1; y = y + 1; w = w + 1; z = z + 10;"] , "(x >= 4)") , 
            ( ["x = x + 1; y = y + 100; w = w + 1; z = z + 10;" , "y = -y; w = w + 1; z = z + 10;"] , "(y > 10*w) && (z >= 100*x) && !(x >= 4)") , 
            ( ["x = x + 1; y = y + 100; w = w + 1; z = z + 10;" , "w = w + 1; z = z + 10;"] , "!((x >= 4) || ((y > 10*w) && (z >= 100*x) && !(x >= 4)))")]
        Q = "(x >= 4) && (y <= 2)"

    class gj2007:
        V = "int x,y;"
        P = "(x == 0) && (y == 50)"
        B = "(x < 100)"
        T = [ ( ["x = x + 1;"] , "(x < 50)") , ( ["x = x + 1;y = y + 1;"] , "!(x < 50)" ) ]
        Q = "(y == 100)"    

    class gj2007b:
        V = "int x,m,n;"
        P = "(x == 0) && (m == 0)"
        B = "(x < n)"
        T = [ ( ["m = x", "x = x + 1"] , "") ]
        Q = "((m >= 0) && (m < n)) || (n <= 0)"

    class gr2006:
        V = "int x,y;"
        P = "(x == 0) && (y == 0)"
        B = "!(((x < 50) && (y < -1)) || ((x >= 50) && (y < 1)))"
        T = [ ( ["y = y + 1; x = x + 1;"] , "(x < 50) && (y >= 0)") , ( ["y = y - 1; x = x + 1;"] , "(x >= 50) && (y >= 0)") ]
        Q = "(x == 100)"    

    # gsv2008.c.i.p+cfa-reducer has same IR as gsv2008
    # gsv2008.c.i.v+cfa-reducer.c has same IR as gsv2008
    # gsv2008.c.i.v+lhb-reducer.c has same IR as gsv2008
    class gsv2008:
        V = "int x,y;"
        P = "((x == -50) && (y > -1000) && (y < 1000000))"
        B = "(x < 0)"
        T = [ ( ["x = x + y; y= y + 1;"] , "") ]
        Q = "(y > 0)"    

    class hhk2008:
        V = "int a, b, res, cnt;"
        P = "(res == a) && (cnt == b) && (a <= 1000000) && (b >= 0) && (b <= 1000000)"
        B = "(cnt > 0)"
        T = [ ( ["cnt = cnt - 1; res = res + 1;"] , "") ]
        Q = "(res == a + b)"

    # jm2006.c.i.v+cfa-reducer has same IR as jm2006
    class jm2006:
        V = "int i, j, x, y;"
        P = "(i >= 0) && (j >= 0) && (x == i) && (y == j)"
        B = "(x > 0) || (x < 0)"
        T = [ ( ["x = x - 1; y = y - 1;"] , "") ]
        Q = "!(i == j) || (y == 0)"

    class jm2006_variant:
        V = "int i,j,x,y,z"
        P = "(i >= 0) && (i <= 1000000) && (j >= 0) && (x == i) && (y == j) && (z == 0)"
        B = "(x > 0) || (x < 0)"
        T = [ ( ["x = x - 1; y = y - 1; z = z + 1;"] , "") ]
        Q = "!(i == j) || (y == -z)"

    # Includes arrays and pointers.
    class mcmillan2006:
        A = []

class loop_new:
    class count_by_1:
        V = "int i"
        P = "(i == 0)"
        B = "(i < 1000000)"
        T = [ ( ["i = i + 1;"] , "") ]
        Q = "(i == 1000000)"

    class count_by_1_variant:
        V = "int i"
        P = "(i == 0)"
        B = "(i < 1000000)"
        T = [ ( ["i = i + 1;"] , "") ]
        Q = "(i <= 1000000)"

    class count_by_2:
        V = "int i"
        P = "(i == 0)"
        B = "(i < 1000000)"
        T = [ ( ["i = i + 2;"] , "") ]
        Q = "(i == 1000000)"

    class count_by_k:
        V = "int i,k"
        P = "(i == 0) && (k >= 0) && (k <= 10)"
        B = "(i < 1000000*k)"
        T = [ ( ["i = i + k;"] , "") ]
        Q = "(i == 1000000 * k)"

    # Introduces randomized value inside loop, so different random value for each iteration
    class count_by_nondet:
        A = []

    # Quadratic Post Condition
    # gauss_sum.i.p+cfa-reducer has same IR as gauss_sum
    # gauss_sum.i.p+lhb-reducer has same IR as gauss_sum
    # gauss_sum.i.v+cfa-reducer has same IR as gauss_sum
    class gauss_sum:
        A = []

    # Contains the modulo operator
    class half:
        A = []

    # Contains double nested loop
    class nested_1:
        A = []


class loop_simple:
    # Contains 5 nested loop
    class deep_nested:
        A = []
    
    # nested_1b has teh same IR as nested_1
    class nested_1:
        V = "int a "
        P = "(a = 0)"
        B = "(a < 6)"
        T = [ ( ["a = a + 1;"] , "") ]
        Q = "(a == 6)"

    # Contains 2 nested loop
    class nested_2:
        A = []

    # Contains 3 nested loop
    class nested_3:
        A = []

    # Contains 4 nested loop
    class nested_4:
        A = []

    # Contains 5 nested loop
    class nested_5:
        A = []

    # Contains 6 nested loop
    class nested_6:
        A = []

class loop_zilu:
    class benchmark01_conjunctive:
        V = "int x,y"
        P = "(x == 1) && (y == 1)"
        B = ""
        T = [ ( ["x= x+y; y= x;"] , "") ]
        Q = "(y >= 1)"

    class benchmark02_linear:
        V = "int n,i,l"
        P = "(l > 0) && (i == l)"
        B = "(i < n)"
        T = [ ( ["i = i + 1;"] , "") ]
        Q = "(l >= 1)"

    class benchmark03_linear:
        V = "int x,y, i, j, flag"
        P = "(x == 0) && (y == 0) && (j == 0) && (i == 0)"
        B = "(flag < 0) || (flag > 0)"
        T = [ ( ["x = x + 1; y = y + 1; i = i + x; j = j + y;"] , "(flag == 0)") , ( ["x = x + 1; y = y + 1; i = i + x; j = j + y+1;"] , "(flag < 0) || (flag > 0)") ]
        Q = "(j >= i)"        

    class benchmark04_conjunctive:
        V = "int k, j, n"
        P = "(n >= 1) && (k >= n) && (j == 0)"
        B = "(j <= n-1)"
        T = [ ( ["j = j + 1; k = k - 1;"] , "") ]
        Q = "(k >= 0)"

    class benchmark05_conjunctive:
        V = "int x,y, n"
        P = "(x>=0) && (x<=y) && (y<n)"
        B = "(x<n)"
        T = [ ( ["x = x + 1;"] , "(x<=y)") , ( ["x = x + 1; y = y + 1;"] , "(x > y)") ]
        Q = "(y==n)"

    class benchmark06_conjunctive:
        V = "int i, j, x, y, k;"
        P = "(x+y==k) && (j==0)"
        B = ""
        T = [ ( ["x = x + 1; y = y - 1;"] , "(j==i)"), ( ["x = x - 1; y = y + 1;"] , "(j< i) || (i < j)") ]
        Q = "(x+y==k)"

    class benchmark07_linear:
        V = "int i, n, k, flag;"
        P = "(n > 0) && (n < 10) && ( (flag == 0) || (flag == 1) )"
        B = "(i<n)"
        T = [ ( ["i = i + 1; k = k + 4000;"] , "(flag == 1)"), ( ["i = i + 1; k = k + 2000;"] , "(flag == 0)") ]
        Q = "(k>n)"    

    class benchmark08_conjunctive:
        V = "int n, sum, i;"
        P = "(n >= 0) && (sum == 0) && (i == 0)"
        B = "(i<n)"
        T = [ ( ["sum = sum + i; i = i + 1;"] , "" ) ]
        Q = "(sum >= 0)"   

    class benchmark09_conjunctive:
        V = "int x,y;"
        P = "(x == y) && (y >=0)"
        B = "((x < 0) || (x > 0) ) && ((y < 0) || (y > 0))"
        T = [ ( ["x = x - 1; y = y - 1;"] , "") ]
        Q = "(y == 0)"

    class benchmark10_conjunctive:
        V = "int i, c;"
        P = "(c == 0) && (i == 0)"
        B = "(i<100) && (i > -1)"
        T = [ ( ["c=c+i; i=i+1;"] , "") ]
        Q = "(c>=0)"

    class benchmark11_linear:
        V = "int x, n;"
        P = "(x == 0) && (n > 0)"
        B = "(x < n )"
        T = [ ( ["x = x + 1;"] , "") ]
        Q = "(x == n)"

    class benchmark12_linear:
        V = "int x,y,t;"
        P = "((x > y) || (x < y)) && (y == t)"
        B = ""
        T = [ ( ["y = y+x;"] , "( x > 0 )") ]
        Q = "(y >= t)"

    class benchmark13_conjunctive:
        V = "int i,j,k;"
        P = "(i==0) && (j==0)"
        B = "(i <= k)"
        T = [ ( ["i = i + 1 ; j = j + 1;"] , "") ]
        Q = "(j==i)"

    class benchmark14_linear:
        V = "int i;"
        P = "(i>=0) && (i<=200)"
        B = "(i>0)"
        T = [ ( ["i = i - 1;"] , "") ]
        Q = "(i>=0)"

    class benchmark15_conjunctive:
        V = "int low, mid, high;"
        P = "(low == 0) && (mid >= 1) && (high == 2*mid)"
        B = "(mid > 0)"
        T = [ ( ["    low = low + 1; high = high - 1;mid = mid - 1;"] , "") ]
        Q = "(low == high)"

    class benchmark16_conjunctive:
        V = "int i,k;"
        P = "(0 <= k) && (k <= 1) && (i == 1)"
        B = ""
        T = [ ( ["i = i + 1; k = k - 1;"] , "") ]
        Q = "(1 <= i + k) && (i + k <= 2) && (i >= 1)"

    class benchmark17_conjunctive:
        V = "int i,k,n;"
        P = "(i==0) && (k==0)"
        B = "(i<n)"
        T = [ ( ["    i = i + 1; k = k + 1;"] , "") ]
        Q = "(k>=n)"

    class benchmark18_conjunctive:
        V = "int i,k,n;"
        P = "((i==0) && (k==0) && (n>0))"
        B = "(i < n)"
        T = [ ( ["i = i + 1; k = k + 1;"] , "") ]
        Q = "(i == k) && (k == n)"

    class benchmark19_conjunctive:
        V = "int j,k,n;"
        P = "((j==n) && (k==n) && (n>0))"
        B = "(j>0) && (n>0)"
        T = [ ( ["j = j - 1; k = k - 1;"] , "") ]
        Q = "(k == 0)"

    class benchmark20_conjunctive:
        V = "int i,n, sum;"
        P = "(i==0) && (n>=0) && (n<=100) && (sum==0)"
        B = "(i<n)"
        T = [ ( ["sum = sum + i; i = i + 1;"] , "") ]
        Q = "(sum>=0)"        

    class benchmark21_disjunctive:
        V = "int x,y;"
        P = "(y>0) || (x>0)"
        B = "(x+y <= -2)"
        T = [ ( ["x = x + 1;"] , "(x > 0)") , ( ["y = y + 1;"] , "(x <= 0)") ]
        Q = "(y>0) || (x>0)"

    class benchmark23_conjunctive:
        V = "int i,j;"
        P = "(i==0) && (j==0)"
        B = "(i<100)"
        T = [ ( ["j = j + 2; i = i + 2;"] , "") ]
        Q = "(j==200)"

    class benchmark24_conjunctive:
        V = "int i,k,n;"
        P = "(i==0) && (k==n) && (n>=0)"
        B = "(i<n)"
        T = [ ( ["k = k - 1; i = i + 2;"] , "") ]
        Q = "(2*k>=n-1)"

    class benchmark25_linear:
        V = "int x;"
        P = "(x<0)"
        B = "(x<10)"
        T = [ ( ["x=x+1;"] , "") ]
        Q = "(x==10)"

    class benchmark26_linear:
        V = "int x,y;"
        P = "(x<y)"
        B = "(x<y)"
        T = [ ( ["x=x+1;"] , "") ]
        Q = "(x==y)"

    class benchmark27_linear:
        V = "int i,j,k"
        P = "(i<j) && (k>i-j)"
        B = "(i<j)"
        T = [ ( ["    k=k+1;i=i+1;"] , "") ]
        Q = "(k > 0)"


    # (* program semantics simplified, but same program)
    class benchmark28_linear:
        V = "int i,j;"
        P = "((i-j < 0) && (i + j > 0)) || ((i-j > 0) && (i + j < 0))"
        B = "(i < j)"
        T = [ ( ["i = j - i; j = j - i;"] , "(j < 2*i)") , ( ["j = j - i;"] , "(j >= 2*i)") ]
        Q = "(j == i)"

    class benchmark29_linear:
        V = "int x,y;"
        P = "(x<y)"
        B = "(x<y)"
        T = [ ( ["x=x+100;"] , "") ]
        Q = "(x >= y && x <= y + 99)"

    class benchmark30_conjunctive:
        V = "int x,y;"
        P = "(y == x)"
        B = ""
        T = [ ( ["    x = x + 1; y = y + 1;"] , "") ]
        Q = "(x == y)"

    class benchmark31_disjunctive:
        V = "int x,y;"
        P = "(x < 0)"
        B = "(x < 0)"
        T = [ ( ["x=x+y; y = y + 1;"] , "") ]
        Q = "(y>=0)"

    class benchmark32_linear:
        V = "int x;"
        P = "(x==1) || (x==2)"
        B = ""
        T = [ ( ["x= 2;"] , "(x == 1)") ,  ( ["x = 1;"] , "(x == 2)") ]
        Q = "(x<=8)"

    class benchmark33_linear:
        V = "int x;"
        P = "(x>=0)"
        B = "(x<100 && x>=0)"
        T = [ ( ["x = x + 1;"] , "") ]
        Q = "(x>=100)"

    class benchmark34_conjunctive:
        V = "int j,k,n;"
        P = "((j==0) && (k==n) && (n>0))"
        B = "(j<n) && (n>0)"
        T = [ ( ["j = j + 1;k = k - 1;"] , "") ]
        Q = "(k == 0)"

    class benchmark35_linear:
        V = "int x;"
        P = "(x>=0)"
        B = "((x>=0) && (x<10))"
        T = [ ( ["x=x+1;"] , "") ]
        Q = "(x>=10)"

    class benchmark36_conjunctive:
        V = "int x,y;"
        P = "(x == y) && (y == 0)"
        B = ""
        T = [ ( ["x = x + 1; y = y + 1;"] , "") ]
        Q = "(x == y) && (x >= 0)"

    class benchmark37_conjunctive:
        V = "int x,y;"
        P = "(x == y) && (x >= 0)"
        B = "(x > 0)"
        T = [ ( ["x = x - 1; y = y - 1;"] , "") ]
        Q = "(y >= 0)"

    class benchmark38_conjunctive:
        V = "int x,y;"
        P = "(x == y) && (y == 0)"
        B = ""
        T = [ ( ["x = x +4; y = y + 1;"] , "") ]
        Q = "(x == 4*y) && (x >= 0)"

    class benchmark39_conjunctive:
        V = "int x,y;"
        P = "(x == 4*y && x >= 0)"
        B = "(x > 0)"
        T = [ ( ["    x = x - 4; y = y - 1;"] , "") ]
        Q = "(y>=0)"

    # (* program semantics simplified, but same program)
    class benchmark40_polynomial:
        V = "int x,y;"
        P = "((x >= 0) && (y >= 0)) || ((x <= 0) && (y <= 0))"
        B = ""
        T = [ ( ["y = y + 1;"] , "(x > 0)") , ( ["x = x - 1;"] , "(x < 0)") , ( ["x = x + 1;"] , "(x == 0) && (y > 0)") , ( ["x = x - 1;"] , "(x == 0) && (y <= 0)") ]
        Q = "((x >= 0) && (y >= 0)) || ((x <= 0) && (y <= 0))"

    class benchmark41_conjunctive:
        V = "int x,y,z;"
        P = "(x == y) && (y == 0) && (z==0)"
        B = ""
        T = [ ( ["x = x + 1; y = y + 1; z = z - 2;"] , "") ]
        Q = "(x == y) && (x >= 0) && (x+y+z==0)"

    class benchmark42_conjunctive:
        V = "int x,y,z;"
        P = "(x == y && x >= 0 && x+y+z==0)"
        B = "(x > 0)"
        T = [ ( ["    x = x - 1; y = y - 1; z = z + 2;"] , "") ]
        Q = "(z<=0)"

    class benchmark43_conjunctive:
        V = "int x,y;"
        P = "(x < 100) && (y < 100)"
        B = "(x < 100) && (y < 100)"
        T = [ ( ["x=x+1 ; y=y+1;"] , "") ]
        Q = "(x == 100) || (y == 100)"

    class benchmark44_disjunctive:
        V = "int x,y;"
        P = "(x<y)"
        B = "(x<y)"
        T = [ ( ["x = x+7; y = y-10;"] , "((x < 0) && (y < 0))") , ( ["x = x + 7; y = y + 3;"] , "((x < 0) && (y >= 0))") , ( ["x = x + 10; y = y + 3;"] , "(x >= 0)")  ]
        Q = "(x >= y) && (x <= y + 16)"

    class benchmark45_disjunctive:
        V = "int x,y;"
        P = "(y>0) || (x>0)"
        B = ""
        T = [ ( ["x = x + 1;"] , "(x > 0)") , ( ["y = y + 1;"] , "(x <= 0)") ]
        Q = "(x>0 || y>0)"

    class benchmark46_disjunctive:
        V = "int x,y,z;"
        P = "(y>0) || (x>0) || (z>0)"
        B = ""
        T = [ ( ["x = x + 1;"] , "(x > 0)") , ( ["y = y + 1;"] , "(x <= 0) && (y>0)") , ( ["z = z + 1;"] , "(x <= 0) && (y <=0)") ]
        Q = "(y>0) || (x>0) || (z>0)"

    class benchmark47_linear:
        V = "int x,y;"
        P = "(x<y)"
        B = "(x<y)"
        T = [ ( ["x = x + 7; y = y - 10;"] , "(x < 0) && (y < 0)") , ( ["x = x + 10; y = y - 10;"] , "(x >= 0) && (y < 0)")
              , ( ["x = x + 7; y = y + 3;"] , "(x < 0) && (y >= 0)") , ( ["x = x + 10; y = y + 3;"] , "(x >= 0) && (y >= 0)")  ]
        Q = "(x >= y) && (x <= y + 16)"

    class benchmark48_linear:
        V = "int i,j,k;"
        P = "(i < j) && (k > 0)"
        B = "(i < j)"
        T = [ ( ["k = k+1 ; i = i+1;"] , "") ]
        Q = "(k > j - i)"

    class benchmark49_linear:
        V = "int i,j,r;"
        P = "(r > i + j)"
        B = "(i > 0)"
        T = [ ( ["i = i - 1; j = j + 1;"] , "") ]
        Q = "(r > i + j)"

    class benchmark50_linear:
        V = "int xa, ya;"
        P = "(xa + ya > 0)"
        B = "(xa > 0)"
        T = [ ( ["    xa = xa - 1; ya = ya + 1;"] , "") ]
        Q = "(ya >= 0)"

    class benchmark51_polynomial:
        V = "int x;"
        P = "((x>=0) && (x<=50))"
        B = ""
        T = [ ( ["x = x + 1;"] , "((x == 0) || (x > 50))") , ( ["x = x - 1;"] , "((x > 0) && (x <= 50)) || (x < 0)") ]
        Q = "((x>=0) && (x<=50))"

    # modified, but equivalent program
    class benchmark52_polynomial:
        V = "int i;"
        P = "((i < 10) && (i > -10))"
        B = "((i < 10) && (i > -10))"
        T = [ ( ["i = i + 1;"] , "") ]
        Q = "(i == 10)"

    # modified, but equivalent program
    class benchmark53_polynomial:
        V = "int x,y;"
        P = "(((x >= 0) && (y >= 0)) || ((x <= 0) && (y <= 0)))"
        B = ""
        T = [ ( ["y = y + 1;"] , "(x > 0)") , ( ["x = x - 1;"] , "(x < 0)") , ( ["x = x + 1;"] , "((x == 0) && (y > 0))") , ( ["x = x - 1;"] , "((x == 0) && (y <= 0))") ]
        Q = "(((x >= 0) && (y >= 0)) || ((x <= 0) && (y <= 0)))"

class loops_crafted_1:
    # No invariant exists for this as Q doesn't hold
    class Mono1_1_1:
        V = "int x;"
        P = "(x == 0)"
        B = "(x < 100000000)"
        T = [ ( ["x = x + 1;"] , "(x < 10000000)") , ( ["x = x + 2;"] , "(x >= 10000000)") ]
        Q = "(x == 100000001)"

    class Mono1_1_2:
        V = "int x;"
        P = "(x == 0)"
        B = "(x < 100000000)"
        T = [ ( ["x = x + 1;"] , "(x < 10000000)") , ( ["x = x + 2;"] , "(x >= 10000000)") ]
        Q = "(x == 100000000)"

    class Mono3_1:
        V = "int x,y;"
        P = "((x == 0) && (y == 0))"
        B = "(x < 1000000)"
        T = [ ( ["y = y + 1; x = x + 1;"] , "(x < 500000)") , ( ["y = y - 1; x = x + 1;"] , "(x >= 500000)") ]
        Q = "((y > 0) || (y < 0))"

    class Mono4_1:
        V = "int x,y;"
        P = "((x == 0) && (y == 500000))"
        B = "(x < 1000000)"
        T = [ ( ["x = x + 1;"] , "(x < 500000)") , ( ["y = y + 1; x = x + 1;"] , "(x >= 500000)") ]
        Q = "( x < y) || ( y < x)"


    class name:
        V = ""
        P = ""
        B = ""
        T = [ ( [""] , "") ]
        Q = ""

class_wrapper_program_converter(loops_crafted_1.Mono4_1)

