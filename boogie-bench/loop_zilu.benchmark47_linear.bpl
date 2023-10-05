
function {:existential true} inv(x: int, y: int): bool;
procedure main()
{
  var x, y: int;
  
  assume (x*1)+(y*-1)<0;
  while ((x*1)+(y*-1)<0)
  invariant inv(x, y);
  {
    
    
    if ((x*1)+(y*0)<0 && (x*0)+(y*1)<0) {
        x := 1*x+0*y+7;
y := 0*x+1*y+-10;

    }

    if ((x*1)+(y*0)>=0 && (x*0)+(y*1)<0) {
        x := 1*x+0*y+10;
y := 0*x+1*y+-10;

    }

    if ((x*1)+(y*0)<0 && (x*0)+(y*1)>=0) {
        x := 1*x+0*y+7;
y := 0*x+1*y+3;

    }

    if ((x*1)+(y*0)>=0 && (x*0)+(y*1)>=0) {
        x := 1*x+0*y+10;
y := 0*x+1*y+3;

    }

  }
  assert (x*1)+(y*-1)>=0 && (x*1)+(y*-1)<=16;
}
