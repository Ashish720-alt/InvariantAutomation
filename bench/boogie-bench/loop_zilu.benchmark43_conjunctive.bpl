
function {:existential true} inv(x: int, y: int): bool;
procedure main()
{
  var x, y: int;
  var b0: bool;
  assume (x*1)+(y*0)<100 && (x*0)+(y*1)<100;
  while ((x*1)+(y*0)<100 && (x*0)+(y*1)<100)
  invariant inv(x, y);
  {
    havoc b0;
    
    if (b0) {
        x := 1*x+0*y+1;
y := 0*x+1*y+1;

    }

  }
  assert (x*1)+(y*0)==100;
}
