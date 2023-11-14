
function {:existential true} inv(x: int): bool;
procedure main()
{
  var x: int;
  var b: bool;
  assume (x*1)==1 || (x*1)==2;
  while (b)
  invariant inv(x);
  {
    havoc b;
    
    if ((x*1)==1) {
        x := 0*x+2;

    }

    if ((x*1)==2) {
        x := 0*x+1;

    }

    if ((x*1)<=0 || (x*1)>=3) {
        x := 1*x+0;

    }

  }
  assert (x*1)<=8;
}
