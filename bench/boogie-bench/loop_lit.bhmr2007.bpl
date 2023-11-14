
function {:existential true} inv(a: int, b: int, i: int, n: int): bool;
procedure main()
{
  var a, b, i, n: int;
  var b0: bool;var b1: bool;
  assume (a*1)+(b*0)+(i*0)+(n*0)==0 && (a*0)+(b*1)+(i*0)+(n*0)==0 && (a*0)+(b*0)+(i*1)+(n*0)==0 && (a*0)+(b*0)+(i*0)+(n*1)>=0 && (a*0)+(b*0)+(i*0)+(n*1)<=1000000;
  while ((a*0)+(b*0)+(i*1)+(n*-1)<0)
  invariant inv(a, b, i, n);
  {
    havoc b0;havoc b1;
    
    if (b0) {
        a := 1*a+0*b+0*i+0*n+1;
b := 0*a+1*b+0*i+0*n+2;
i := 0*a+0*b+1*i+0*n+1;
n := 0*a+0*b+0*i+1*n+0;

    }

    if (b1) {
        a := 1*a+0*b+0*i+0*n+2;
b := 0*a+1*b+0*i+0*n+1;
i := 0*a+0*b+1*i+0*n+1;
n := 0*a+0*b+0*i+1*n+0;

    }

  }
  assert (a*1)+(b*1)+(i*0)+(n*-3)==0;
}
