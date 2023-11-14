
function {:existential true} inv(x: int, y: int): bool;
procedure main()
{
  var x, y: int;
  
  assume (x*1)+(y*0)==0 && (x*0)+(y*1)==50;
  while ((x*1)+(y*0)<100)
  invariant inv(x, y);
  {
    
    
    if ((x*1)+(y*0)<50) {
        x := 1*x+0*y+1;
y := 0*x+1*y+0;

    }

    if ((x*1)+(y*0)>=50) {
        x := 1*x+0*y+1;
y := 0*x+1*y+1;

    }

  }
  assert (x*0)+(y*1)==100;
}
