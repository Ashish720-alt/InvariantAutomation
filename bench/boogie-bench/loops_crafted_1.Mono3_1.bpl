
function {:existential true} inv(x: int, y: int): bool;
procedure main()
{
  var x, y: int;
  
  assume (x*1)+(y*0)==0 && (x*0)+(y*1)==0;
  while ((x*1)+(y*0)<1000000)
  invariant inv(x, y);
  {
    
    
    if ((x*1)+(y*0)<500000) {
        x := 1*x+0*y+1;
y := 0*x+1*y+1;

    }

    if ((x*1)+(y*0)>=500000) {
        x := 1*x+0*y+1;
y := 0*x+1*y+-1;

    }

  }
  assert (x*0)+(y*1)>0;
}
