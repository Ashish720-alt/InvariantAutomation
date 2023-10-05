
function {:existential true} inv(x: int, cnt: int, a: int): bool;
procedure main()
{
  var x, cnt, a: int;
  var b0: bool;
  assume (x*1)+(cnt*0)+(a*-1)==0 && (x*0)+(cnt*1)+(a*0)==1 && (x*0)+(cnt*0)+(a*1)>=0;
  while ((x*0)+(cnt*1)+(a*0)>0)
  invariant inv(x, cnt, a);
  {
    havoc b0;
    
    if (b0) {
        x := 1*x+0*cnt+0*a+1;
cnt := 0*x+1*cnt+0*a+-1;
a := 0*x+0*cnt+1*a+0;

    }

  }
  assert (x*1)+(cnt*0)+(a*-1)==1;
}
