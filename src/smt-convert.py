# Translate matrix representation to SMT-LIB2 format
# Remember to modify `repr.py` to change the representation of transition matrix before using this script


from input import *
import argparse
import os
import numpy as np


def convert_op(op: int) -> str:
    # -2: <, 2: >, 0: ==, -1: <=, 1: >=
    match op:
        case -2:
            return "<"
        case 2:
            return ">"
        case 0:
            return "="
        case -1:
            return "<="
        case 1:
            return ">="
        case _:
            raise Exception("Unknown operator " + str(op))


# rhs is of shape (len(var)), which are coeffs on each variable
def convert_rhs(rhs: np.ndarray, var: list) -> str:
    if not (len(rhs.shape) == 1 and rhs.shape[0] == len(var)):
        raise Exception("RHS shape is not correct")

    # remove 0 coefficients
    rhs = list(rhs)
    z = [(rhs[i], var[i]) for i in range(len(rhs)) if rhs[i] != 0]
    if len(z) == 0:
        return "0"
    else:
        rhs, var = zip(*z)

    var_n = len(var)
    rhs = [f"(* {str(rhs[i])} {var[i]})" if rhs[i] !=
           1 else f"{var[i]}" for i in range(var_n)]

    def rec(rhs):
        if len(rhs) == 1:
            return rhs[0]
        return f"(+ {rhs[0]} {rec(rhs[1:])})"
    return rec(rhs)

# cond is of shape (disjunctive, conjunctive, var_n + 2)


def convert_pred(cond: list[np.ndarray], var: list) -> str:
    if cond is None:
        return None
    var_n = len(var)

    # pred is of shape (var_n + 2)
    def convert_clause(pred: np.ndarray) -> str:
        return f"({convert_op(pred[-2])} {convert_rhs(pred[:-2], var)} {str(pred[-1])})"

    def convert_conj(conj: np.ndarray) -> str:
        if not (len(conj.shape) == 2 and conj.shape[-1] == var_n + 2):
            raise Exception("Conjunctive matrix shape is not correct")
        if conj.shape[0] == 1:
            return f"{convert_clause(conj[0])}"
        return f"(and {" ".join([convert_clause(conj[i]) for i in range(conj.shape[0])])})"

    if len(cond) == 1:
        return f"{convert_conj(cond[0])}"
    return f"(or {" ".join([convert_conj(i) for i in cond])})"

# len(var) == len(var_p) == var_n
# trans is of shape (var_n + 1, var_n + 1)


def convert_trans(trans: np.ndarray, var: list, var_p: list) -> str:
    var_n = len(var)
    var_np = len(var_p)
    (n, m) = trans.shape
    if not (n == var_n + 1 and m == var_n + 1):
        raise Exception("Transition matrix shape is not correct")
    if not (n == var_np + 1 and m == var_np + 1):
        raise Exception("Transition matrix shape is not correct")
    return f"""(and {" ".join([f"(= {var_p[i]} {f"(+ {convert_rhs(trans[i][:-1], var)} {trans[i][-1]})" if trans[i][-1] != 0 else convert_rhs(trans[i][:-1], var)})" for i in range(var_n)])})"""


def convert_inv(inv: str, vars: list[str]) -> str:
    return f"({inv} {" ".join(vars)})"


def convert(i) -> str:
    # decompose input
    vars, p, tlist, q = i.Var, np.array(i.P), i.T, np.array(i.Q)
    b = np.array(i.B) if i.B is not None else None
    trans, cond = [], []
    print(tlist)
    for t in tlist:
        # print(t)
        (_trans, _cond) = (np.array(t[0]), t[1])
        (dnf, _, _) = _trans.shape
        for i in range(dnf):
            trans.append(_trans[i])
            cond.append(_cond)

    # validate input
    var_n = len(vars)
    if p.shape[-1] != var_n + 2 or len(p.shape) != 3:
        raise Exception(
            "P matrix shape is not correct for input " + i.__class__.__name__
        )
    if b is not None and (b.shape[-1] != var_n + 2 or len(b.shape) != 3):
        raise Exception(
            "B matrix shape is not correct for input " + i.__class__.__name__
        )
    if q.shape[-1] != var_n + 2 or len(q.shape) != 3:
        raise Exception(
            "Q matrix shape is not correct for input " + i.__class__.__name__
        )
    for t, c in zip(trans, cond):
        if t.shape[-1] != var_n + 1:
            raise Exception(
                "T's transition matrix shape is not correct for input "
                + i.__class__.__name__
            )
        if c is None:
            continue
        for c in c:
            if c.shape[-1] != var_n + 2:
                raise Exception(
                    "T's condition matrix shape is not correct for input "
                    + i.__class__.__name__
                )

    # generate boogie program
    inv_name = "inv"
    vars_p = [v + "p" for v in vars]
    P = convert_pred(p, vars)
    Q = convert_pred(q, vars)
    B = convert_pred(b, vars)
    I_x = convert_inv(inv_name, vars)
    I_xp = convert_inv(inv_name, vars_p)
    assert_lhs = f"(and (not {B}) {I_x})" if b is not None else f"{I_x}"
    return f"""
    (declare-fun |{inv_name}| ({" ".join(["Int" for _ in vars])}) Bool)
    (assert 
      (forall ({' '.join(map(lambda v: f"({v} Int)", vars + vars_p))}) 
        (and 
          (=> {P} {I_x})
          {"\n".join(map(lambda t:
                         (f"(=> {convert_pred(t[1], vars)}" if t[1] is not None else f"") +
                         (f"(=> (and {I_x} {B} {convert_trans(t[0], vars, vars_p)}) {I_xp})"
                         if B is not None else f"(=> (and {I_x} {convert_trans(t[0], vars, vars_p)}) {I_xp})") +
                         (")" if t[1] is not None else ""),
                         zip(trans, cond)))}
          (=> {assert_lhs} {Q})
        )
      )
    )
    (check-sat)
    (get-model)
    """


def output_to_file(output: str, filename: str):
    with open(filename, "w") as f:
        f.write(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCMC Invariant Search")
    parser.add_argument("-i", "--input", type=str, help="Input object name")
    parser.add_argument("-o", "--output", type=str, help="Output file name")
    parser.add_argument("-a", "--all", action="store_true",
                        help="Run all inputs")
    parser.add_argument("-f", "--folder", type=str, help="Output folder name")
    parse_res = vars(parser.parse_args())

    if parse_res["all"]:
        if parse_res["folder"] is None:
            print(parser.print_help())
            print("Please specify the output folder.")
            exit(1)
        output_folder = parse_res["folder"]
        for subfolder in Inputs.__dict__:
            if subfolder[0] != "_":
                for inp in getattr(Inputs, subfolder).__dict__:
                    if inp[0] != "_":
                        # try:
                        print(inp)
                        output_to_file(
                            convert(
                                getattr(getattr(Inputs, subfolder), inp)),
                            os.path.join(
                                output_folder,
                                str(subfolder) + "." + str(inp) + ".smt",
                            ),
                        )
                        # except Exception as e:
                        #     print(
                        #         "Error when converting "
                        #         + str(subfolder)
                        #         + "."
                        #         + str(inp)
                        #         + ": "
                        #         + str(e)
                        # )
    else:
        if parse_res["input"] is None or parse_res["output"] is None:
            print(parser.print_help())
            print("Please specify input case name and output file name.")
            exit(1)
        (first_name, last_name) = parse_res["input"].split(".")
        for subfolder in Inputs.__dict__:
            if subfolder == first_name:
                for inp in getattr(Inputs, subfolder).__dict__:
                    if inp == last_name:
                        output_to_file(
                            convert(getattr(getattr(Inputs, subfolder), inp)),
                            parse_res["output"],
                        )
