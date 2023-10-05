
function {:existential true} inv(xa: int, ya: int): bool;
procedure main()
{
  var xa, ya: int;
  var b0: bool;
  assume (xa*1)+(ya*1)>0;
  while ((xa*1)+(ya*0)>0)
  invariant inv(xa, ya);
  {
    havoc b0;
    
    if (b0) {
        xa := 1*xa+0*ya+-1;
ya := 0*xa+1*ya+1;

    }

  }
  assert (xa*0)+(ya*1)>=0;
}
