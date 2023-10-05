
function {:existential true} inv(x: int): bool;
procedure main()
{
  var x: int;
  var b0: bool;
  assume x>=0 && x<=50;
  while (b0)
  invariant inv(x);
  {
    havoc b0;

    if (x>50) {
      x := x+1;
    }

    if (x == 0) {
      x:= x+1;
    } else {
      x:= x-1;
    }
  }
  assert x>=0 && x<=50;
}
