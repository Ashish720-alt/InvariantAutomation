
function {:existential true} inv(a: int, b: int, res: int, cnt: int): bool;
procedure main()
{
  var a, b, res, cnt: int;
  var b0: bool;
  assume (a*-1)+(b*0)+(res*1)+(cnt*0)==0 && (a*0)+(b*-1)+(res*0)+(cnt*1)==0 && (a*1)+(b*0)+(res*0)+(cnt*0)<=1000000 && (a*0)+(b*1)+(res*0)+(cnt*0)>=0 && (a*0)+(b*1)+(res*0)+(cnt*0)<=1000000;
  while ((a*0)+(b*0)+(res*0)+(cnt*1)>0)
  invariant inv(a, b, res, cnt);
  {
    havoc b0;
    
    if (b0) {
        a := 1*a+0*b+0*res+0*cnt+0;
b := 0*a+1*b+0*res+0*cnt+0;
res := 0*a+0*b+1*res+0*cnt+1;
cnt := 0*a+0*b+0*res+1*cnt+-1;

    }

  }
  assert (a*-1)+(b*-1)+(res*1)+(cnt*0)==0;
}
