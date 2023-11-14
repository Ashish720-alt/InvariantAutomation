
function {:existential true} inv(x: int): bool;
procedure main()
{
  var x: int;
  
  assume (x*1)==0;
  while ((x*1)<100000000)
  invariant inv(x);
  {
    
    
    if ((x*1)<10000000) {
        x := 1*x+1;

    }

    if ((x*1)>=10000000) {
        x := 1*x+2;

    }

  }
  assert (x*1)==100000001;
}
