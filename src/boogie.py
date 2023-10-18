# Translate matrix representation to a Boogie program

from input import *
import argparse
import os
import numpy as np

transition_template = """
    if ({cond}) {{
        {trans}
    }}
"""

prog_template = """
function {{:existential true}} inv({list_var_type}): bool;
procedure main()
{{
  var {list_var}: int;
  {B_decl}
  assume {P};
  while ({B})
  invariant inv({list_var});
  {{
    {B_havoc}
    {T}
  }}
  assert {Q};
}}
"""


def convert_op(op: int) -> str:
    # -2: <, 2: >, 0: ==, -1: <=, 1: >=
    match op:
        case -2:
            return "<"
        case 2:
            return ">"
        case 0:
            return "=="
        case -1:
            return "<="
        case 1:
            return ">="
        case _:
            raise Exception("Unknown operator " + str(op))


def convert_sum(sum: np.ndarray, var: list) -> str:
    output = []
    for i in range(len(sum)):
        if sum[i] == 1:
            output.append(var[i])
        if sum[i] < 0:
            output.append("(" + str(sum[i]) + "*" + var[i] + ")")
        if sum[i] != 0:
            output.append(str(sum[i]) + "*" + var[i])
    if output == []:
        return ["0"]
    return output


# cond is of shape (disjunctive, conjunctive, var_n + 2)
def convert_cond(cond: np.ndarray, var: list) -> str:
    var_n = len(var)
    assert len(cond.shape) == 3 and cond.shape[-1] == var_n + 2

    def convert_pred(pred: np.ndarray) -> str:
        lhs = "+".join(convert_sum(pred, var))
        return lhs + convert_op(pred[-2]) + str(pred[-1])

    def convert_conj(conj: np.ndarray) -> str:
        return " && ".join([convert_pred(conj[i]) for i in range(conj.shape[0])])

    return " || ".join([convert_conj(cond[i]) for i in range(cond.shape[0])])


def convert_trans(trans: np.ndarray, var: list) -> str:
    var_n = len(var)
    match trans.shape:
        case (n, m) if n == var_n + 1 and m == var_n + 1:
            pass
        case _:
            raise Exception("Transition matrix shape is not correct")

    return (
        ";\n".join(
            [
                var[i]
                + " := "
                + "+".join(convert_sum(trans[i], var))
                + "+"
                + str(trans[i][var_n])
                for i in range(var_n)
            ]
        )
        + ";\n"
    )


def convert(i) -> str:
    # decompose input
    var, p, b, tlist = i.Var, np.array(i.P), np.array(i.B), i.T
    q = i.Q[0][np.newaxis, :, :]
    trans, cond = [], []
    for t in tlist:
        (_trans, _cond) = (np.array(t[0]), np.array(t[1]))
        trans.append(_trans)
        cond.append(_cond)

    # validate input
    var_n = len(var)
    if p.shape[-1] != var_n + 2 or len(p.shape) != 3:
        raise Exception(
            "P matrix shape is not correct for input " + i.__class__.__name__
        )
    if b.shape[-1] != var_n + 2 or len(b.shape) != 3:
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
        if c.shape[-1] != var_n + 2:
            raise Exception(
                "T's condition matrix shape is not correct for input "
                + i.__class__.__name__
            )

    # generate boogie program
    list_var = ", ".join(var)
    list_var_type = ", ".join([var[i] + ": int" for i in range(var_n)])
    P, Q, B = convert_cond(p, var), convert_cond(q, var), convert_cond(b, var)
    # print(b)
    # print(np.any(b))
    if not np.any(b):
        B = "b"
        B_decl = "var b: bool;"
        B_havoc = "havoc b;"
    else:
        B_decl, B_havoc = "", ""
    T = ""
    b_count = 0
    for t, c in zip(trans, cond):
        if np.any(c):
            cond = convert_cond(c, var)
        else:
            b_name = "b" + str(b_count)
            b_count += 1
            B_decl += "var " + b_name + ": bool;"
            B_havoc += "havoc " + b_name + ";"
            cond = b_name
        trans = convert_trans(t[0], var)
        T += transition_template.format(cond=cond, trans=trans)

    return prog_template.format(
        list_var_type=list_var_type,
        list_var=list_var,
        P=P,
        Q=Q,
        B=B,
        T=T,
        B_decl=B_decl,
        B_havoc=B_havoc,
    )


def output_to_file(output: str, filename: str):
    with open(filename, "w") as f:
        f.write(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCMC Invariant Search")
    parser.add_argument("-i", "--input", type=str, help="Input object name")
    parser.add_argument("-o", "--output", type=str, help="Output file name")
    parser.add_argument("-a", "--all", action="store_true", help="Run all inputs")
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
                        try:
                            output_to_file(
                                convert(getattr(getattr(Inputs, subfolder), inp)),
                                os.path.join(
                                    output_folder,
                                    str(subfolder) + "." + str(inp) + ".bpl",
                                ),
                            )
                        except Exception as e:
                            print(
                                "Error when converting "
                                + str(subfolder)
                                + "."
                                + str(inp)
                                + ": "
                                + str(e)
                            )
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
