
function {:existential true} inv(x: int): bool;
procedure main()
{
  var x: int;
  var b0: bool;
  assume (x*1)==0;
  while ((x*1)<=5)
  invariant inv(x);
  {
    havoc b0;
    
    if (b0) {
        x := 1*x+1;

    }

  }
  assert (x*1)<=6;
}
