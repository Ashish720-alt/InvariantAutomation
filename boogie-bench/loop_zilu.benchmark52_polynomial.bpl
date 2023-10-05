
function {:existential true} inv(i: int): bool;
procedure main()
{
  var i: int;
  var b0: bool;
  assume (i*1)<10 && (i*1)>-10;
  while ((i*1)<10 && (i*1)>-10)
  invariant inv(i);
  {
    havoc b0;
    
    if (b0) {
        i := 1*i+1;

    }

  }
  assert (i*1)==10;
}
