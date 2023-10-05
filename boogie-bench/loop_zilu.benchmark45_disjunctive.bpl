
function {:existential true} inv(x: int, y: int): bool;
procedure main()
{
  var x, y: int;
  var b: bool;
  assume (x*0)+(y*1)>0 || (x*1)+(y*0)>0;
  while (b)
  invariant inv(x, y);
  {
    havoc b;
    
    if ((x*1)+(y*0)>0) {
        x := 1*x+0*y+1;
y := 0*x+1*y+0;

    }

    if ((x*1)+(y*0)<=0) {
        x := 1*x+0*y+0;
y := 0*x+1*y+1;

    }

  }
  assert (x*1)+(y*0)>0;
}
