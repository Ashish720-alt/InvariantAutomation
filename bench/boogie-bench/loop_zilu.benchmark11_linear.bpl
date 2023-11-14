
function {:existential true} inv(x: int, n: int): bool;
procedure main()
{
  var x, n: int;
  var b0: bool;
  assume (x*1)+(n*0)==0 && (x*0)+(n*1)>0;
  while ((x*1)+(n*-1)<0)
  invariant inv(x, n);
  {
    havoc b0;
    
    if (b0) {
        x := 1*x+0*n+1;
n := 0*x+1*n+0;

    }

  }
  assert (x*1)+(n*-1)==0;
}
