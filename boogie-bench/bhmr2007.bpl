
function {:existential true} b0(a: int, b: int, i: int, n: int): bool;
procedure main()
{
  var a, b, i, n: int;
  assume (a*1)+(b*0)+(i*0)+(n*0)==0 && (a*0)+(b*1)+(i*0)+(n*0)==0 && (a*0)+(b*0)+(i*1)+(n*0)==0 && (a*0)+(b*0)+(i*0)+(n*1)>=0 && (a*0)+(b*0)+(i*0)+(n*1)<=1000000;
  while ((a*0)+(b*0)+(i*1)+(n*-1)<=0)
  invariant b0(a, b, i, n);
  {
    
    if ((a*0)+(b*0)+(i*0)+(n*0)<=0) {
        a := 1*a+0*b+0*i+0*n;
b := 0*a+1*b+0*i+0*n;
i := 0*a+0*b+1*i+0*n;
n := 0*a+0*b+0*i+1*n;

    }

    if ((a*0)+(b*0)+(i*0)+(n*0)<=0) {
        a := 1*a+0*b+0*i+0*n;
b := 0*a+1*b+0*i+0*n;
i := 0*a+0*b+1*i+0*n;
n := 0*a+0*b+0*i+1*n;

    }

  }
  assert (a*1)+(b*1)+(i*0)+(n*-3)==0 || (a*0)+(b*0)+(i*1)+(n*-1)<=0;
}
